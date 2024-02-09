import bz2
import json
import rich
from ortools.sat.python import cp_model

model = cp_model.CpModel()

with bz2.open('data.json.bz2', 'rt') as f:
    data = json.load(f)
    page_data = data['features']['pages'][0]['body']
    rich.print(page_data)

    uniq_tokens = sorted(page_data['tokenPosCount'].keys())
    uniq_token_count = len(uniq_tokens)
    token_idx_vars = []
    for idx in range(page_data['tokenCount']):
        token_idx_choices_vars = []
        for token in uniq_tokens:
            token_idx_choices_vars.append(model.NewBoolVar(f'token_{idx}_{token}'))
        token_idx_vars.append(token_idx_choices_vars)
        # each position must have exactly one token
        model.Add(sum(token_idx_choices_vars) == 1)
    
    # each token must be used exact number of times
    for idx, token in enumerate(uniq_tokens):
        token_vars = [token_idx_vars[i][idx] for i in range(page_data['tokenCount'])]
        pos_sum = sum(page_data['tokenPosCount'][token].values())
        model.Add(sum(token_vars) == pos_sum)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = True
    status = solver.Solve(model)
    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        for idx, token in enumerate(uniq_tokens):
            token_vars = [token_idx_vars[i][idx] for i in range(page_data['tokenCount'])]
            print(f'{token}: {sum(solver.Value(v) for v in token_vars)}')
        # print sequence of tokens
        for tvar_list in token_idx_vars:
            for idx, tvar in enumerate(tvar_list):
                if solver.Value(tvar):
                    print(uniq_tokens[idx], end=' ')
        print()
    else:
        print('No solution found')


