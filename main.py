import bz2
import json
import rich
from ortools.sat.python import cp_model

model = cp_model.CpModel()

class SolutionCounter(cp_model.CpSolverSolutionCallback):
    """Counts the number of solutions."""

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        if self.__solution_count % 1000 == 0:
            print(f'Solution {self.__solution_count} found')

    @property
    def solution_count(self) -> int:
        return self.__solution_count

with bz2.open('data.json.bz2', 'rt') as f:
    data = json.load(f)
    page_data = data['features']['pages'][0]['body']
    rich.print(page_data)

    line_count = page_data['lineCount'] - page_data['emptyLineCount']

    uniq_tokens = sorted(page_data['tokenPosCount'].keys())
    uniq_token_count = len(uniq_tokens)
    token_idx_vars = []
    token_idx_start_vars = []
    token_idx_end_vars = []
    for idx in range(page_data['tokenCount']):
        token_idx_choices_vars = []
        token_idx_choices_start_vars = []
        token_idx_choices_end_vars = []
        for token in uniq_tokens:
            token_var = model.NewBoolVar(f'token_{idx}_{token}')
            token_idx_choices_vars.append(token_var)
            token_start_var = model.NewBoolVar(f'token_{idx}_{token}_is_start')
            token_idx_choices_start_vars.append(token_start_var)
            token_end_var = model.NewBoolVar(f'token_{idx}_{token}_is_end')
            token_idx_choices_end_vars.append(token_end_var)
            model.AddImplication(token_var.Not(), token_start_var.Not())
            model.AddImplication(token_var.Not(), token_end_var.Not())
            model.AddImplication(token_start_var, token_var)
            model.AddImplication(token_end_var, token_var)
        token_idx_vars.append(token_idx_choices_vars)
        token_idx_start_vars.append(token_idx_choices_start_vars)
        token_idx_end_vars.append(token_idx_choices_end_vars)
        # each position must have exactly one token
        model.Add(sum(token_idx_choices_vars) == 1)
    
    # each token must be used exact number of times
    for idx, token in enumerate(uniq_tokens):
        token_vars = [token_idx_vars[i][idx] for i in range(page_data['tokenCount'])]
        pos_sum = sum(page_data['tokenPosCount'][token].values())
        model.Add(sum(token_vars) == pos_sum)

    # for each token, count how many times it is used as a start
    # make sure this count equals beginCharCount
    for begin_char in page_data['beginCharCount']:
        begin_char_count = page_data['beginCharCount'][begin_char]
        start_vars = []
        for token_idx, token in enumerate(uniq_tokens):
            if token[0] == begin_char:
                for idx in range(page_data['tokenCount']):
                    start_vars.append(token_idx_start_vars[idx][token_idx])
        # need to check that we actually found tokens that start/end with this character,
        # because we have seen unicode issues in the data
        # for example: "htid": "penn.ark:/81431/p3k35mf4d"
        if len(start_vars) > 0:
            model.Add(sum(start_vars) == begin_char_count)

    for end_char in page_data['endCharCount']:
        end_char_count = page_data['endCharCount'][end_char]
        end_vars = []
        for token_idx, token in enumerate(uniq_tokens):
            if token[-1] == end_char:
                for idx in range(page_data['tokenCount']):
                    end_vars.append(token_idx_end_vars[idx][token_idx])
        if len(end_vars) > 0:
            model.Add(sum(end_vars) == end_char_count)

    all_start_vars = []
    all_end_vars = []
    for idx in range(page_data['tokenCount']):
        start_vars = [token_idx_start_vars[idx][token_idx] for token_idx in range(uniq_token_count)]
        end_vars = [token_idx_end_vars[idx][token_idx] for token_idx in range(uniq_token_count)]
        all_start_vars.extend(start_vars)
        all_end_vars.extend(end_vars)
    model.Add(sum(all_start_vars) == line_count)
    model.Add(sum(all_end_vars) == line_count)

    solver = cp_model.CpSolver()
    solution_counter = SolutionCounter()
    #solver.parameters.max_time_in_seconds = 10
    #solver.parameters.num_search_workers = 8
    solver.parameters.instantiate_all_variables = False
    solver.parameters.log_search_progress = True
    solver.parameters.enumerate_all_solutions = True
    status = solver.Solve(model, solution_counter)
    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        #for idx, token in enumerate(uniq_tokens):
        #    token_vars = [token_idx_vars[i][idx] for i in range(page_data['tokenCount'])]
        #    print(f'{token}: {sum(solver.Value(v) for v in token_vars)}')
        # print sequence of tokens
        for idx in range(page_data['tokenCount']):
            for token_idx in range(uniq_token_count):
                if solver.Value(token_idx_vars[idx][token_idx]):
                    if solver.Value(token_idx_start_vars[idx][token_idx]):
                        print()
                    print(uniq_tokens[token_idx], end=' ')
    else:
        print('No solution found')
    print()


