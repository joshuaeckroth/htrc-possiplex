import bz2
import json
import rich.progress
import sys
import re
import random
from ortools.sat.python import cp_model
import sentence_transformers
import numpy as np

emb_model = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")

console = rich.get_console()

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

# Load the extracted features
input_file = 'fodorrEXTRACTED.json.bz2'

#make it into different folders and process the bigger text, for more repetitions

with open('fodor.txt') as f:
    original = f.read().strip()

with bz2.open(input_file, 'rt') as f:
    data = json.load(f)
    page_data = data['features']['pages'][0]['body'] #it was 24 not 0 here
    rich.print(page_data)

    line_count = page_data['lineCount'] - page_data['emptyLineCount']

    total_token_count = 0
    for token in page_data['tokenPosCount']:
        total_token_count += sum(page_data['tokenPosCount'][token].values())
    print(f"Total token count: {total_token_count}")

    uniq_tokens = sorted(page_data['tokenPosCount'].keys())
    uniq_token_count = len(uniq_tokens)

    num_random = 100
    rand_txts = []
    rand_embeddings = []
    tokens_with_repeats = []
    for token in page_data['tokenPosCount']:
        tokens_with_repeats += [token for _ in range(sum(page_data['tokenPosCount'][token].values()))]
    for i in range(num_random):
        random.shuffle(tokens_with_repeats)
        rand_txt = " ".join(tokens_with_repeats)
        rand_txt = re.sub(r' ([^a-zA-Z])', r'\1', rand_txt)
        rand_txt = rand_txt.strip()
        rand_txts.append(rand_txt)
        rand_embeddings.append(emb_model.encode(rand_txt))

    num_embeddings = 1000 #CHANGE THIS NUMBER
    embeddings = []
    with rich.progress.Progress() as progress:
        task = progress.add_task("Embedding", total=num_embeddings)
        while len(embeddings) < num_embeddings:
            model = cp_model.CpModel()
            all_bool_vars = []
            token_idx_vars = []
            token_idx_start_vars = []
            token_idx_end_vars = []
            for idx in range(page_data['tokenCount']):
                token_idx_choices_vars = []
                token_idx_choices_start_vars = []
                token_idx_choices_end_vars = []
                for token in uniq_tokens:
                    # did this token appear in this position
                    token_var = model.NewBoolVar(f'token_{idx}_{token}_occurrence')
                    token_idx_choices_vars.append(token_var)
                    all_bool_vars.append(token_var)
                    # each token may be a start or end of a line
                    token_start_var = model.NewBoolVar(f'token_{idx}_{token}_is_start')
                    token_idx_choices_start_vars.append(token_start_var)
                    all_bool_vars.append(token_start_var)
                    token_end_var = model.NewBoolVar(f'token_{idx}_{token}_is_end')
                    token_idx_choices_end_vars.append(token_end_var)
                    all_bool_vars.append(token_end_var)
                    # if token is not planned, it cannot be a start or end
                    model.AddImplication(token_var.Not(), token_start_var.Not())
                    model.AddImplication(token_var.Not(), token_end_var.Not())
                    # if token is a start or end, it must be planned
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
                start_of_text_vars = []
                for token_idx, token in enumerate(uniq_tokens):
                    # first token must be considered a start
                    start_of_text_vars.append(token_idx_start_vars[0][token_idx])
                    if token[0] == begin_char:
                        for idx in range(page_data['tokenCount']):
                            start_vars.append(token_idx_start_vars[idx][token_idx])

                # some token must be a start of text
                model.Add(sum(start_of_text_vars) == 1)
                # need to check that we actually found tokens that start/end with this character,
                # because we have seen unicode issues in the data
                # for example: "htid": "penn.ark:/81431/p3k35mf4d"
                if len(start_vars) > 0:
                    model.Add(sum(start_vars) == begin_char_count)
            # tokens that cannot be begin chars
            for token_idx, token in enumerate(uniq_tokens):
                if token[0] not in page_data['beginCharCount']:
                    for idx in range(page_data['tokenCount']):
                        model.Add(token_idx_start_vars[idx][token_idx] == 0)

            for end_char in page_data['endCharCount']:
                end_char_count = page_data['endCharCount'][end_char]
                end_vars = []
                end_of_text_vars = []
                for token_idx, token in enumerate(uniq_tokens):
                    # last token must be considered an end
                    end_of_text_vars.append(token_idx_end_vars[page_data['tokenCount'] - 1][token_idx])
                    if token[-1] == end_char:
                        for idx in range(page_data['tokenCount']):
                            end_vars.append(token_idx_end_vars[idx][token_idx])
                # some token must be an end of text
                model.Add(sum(end_of_text_vars) == 1)
                if len(end_vars) > 0:
                    model.Add(sum(end_vars) == end_char_count)
            # tokens that cannot be end chars
            for token_idx, token in enumerate(uniq_tokens):
                if token[-1] not in page_data['endCharCount']:
                    for idx in range(page_data['tokenCount']):
                        model.Add(token_idx_end_vars[idx][token_idx] == 0)

            all_start_vars = []
            all_end_vars = []
            for idx in range(page_data['tokenCount']):
                # if idx > 0, if there is a start at idx, there must be an end at idx-1; doesn't matter which token it is for start or end
                if idx > 0:
                    has_start_here_var = model.NewBoolVar(f'has_start_here_{idx}')
                    has_end_here_var = model.NewBoolVar(f'has_end_here_{idx}')
                    here_start_vars = []
                    here_end_vars = []
                    for token_idx in range(uniq_token_count):
                        model.AddImplication(token_idx_start_vars[idx][token_idx], has_start_here_var)
                        model.AddImplication(has_start_here_var.Not(), token_idx_start_vars[idx][token_idx].Not())
                        here_start_vars.append(token_idx_start_vars[idx][token_idx])
                        model.AddImplication(token_idx_end_vars[idx - 1][token_idx], has_end_here_var)
                        model.AddImplication(has_end_here_var.Not(), token_idx_end_vars[idx - 1][token_idx].Not())
                        here_end_vars.append(token_idx_end_vars[idx - 1][token_idx])
                    # if any here_start_vars is true, then has_start_var_here must be true
                    model.Add(has_start_here_var == sum(here_start_vars))
                    model.Add(has_end_here_var == sum(here_end_vars))
                    model.Add(has_start_here_var == has_end_here_var)
                if idx < page_data['tokenCount'] - 1: # if there is an end at idx, there must be a start at idx+1
                    has_start_here_var = model.NewBoolVar(f'has_start_here_{idx}')
                    has_end_here_var = model.NewBoolVar(f'has_end_here_{idx}')
                    here_start_vars = []
                    here_end_vars = []
                    for token_idx in range(uniq_token_count):
                        model.AddImplication(token_idx_start_vars[idx+1][token_idx], has_start_here_var)
                        model.AddImplication(has_start_here_var.Not(), token_idx_start_vars[idx+1][token_idx].Not())
                        here_start_vars.append(token_idx_start_vars[idx+1][token_idx])
                        model.AddImplication(token_idx_end_vars[idx][token_idx], has_end_here_var)
                        model.AddImplication(has_end_here_var.Not(), token_idx_end_vars[idx][token_idx].Not())
                        here_end_vars.append(token_idx_end_vars[idx][token_idx])
                    # if any here_start_vars is true, then has_start_var_here must be true
                    model.Add(has_start_here_var == sum(here_start_vars))
                    model.Add(has_end_here_var == sum(here_end_vars))
                    model.Add(has_start_here_var == has_end_here_var)
                start_vars = [token_idx_start_vars[idx][token_idx] for token_idx in range(uniq_token_count)]
                end_vars = [token_idx_end_vars[idx][token_idx] for token_idx in range(uniq_token_count)]
                all_start_vars.extend(start_vars)
                all_end_vars.extend(end_vars)
            model.Add(sum(all_start_vars) == line_count)
            model.Add(sum(all_end_vars) == line_count)

            if len(embeddings) == 0:
                print("Number of bool vars:", len(all_bool_vars))
            for var in random.choices(all_bool_vars, k=2):
                model.Add(var == random.choice([0, 1]))

            solver = cp_model.CpSolver()
            solution_counter = SolutionCounter()
            solver.parameters.max_time_in_seconds = 10
            solver.parameters.num_search_workers = 8
            #solver.parameters.instantiate_all_variables = False
            solver.parameters.log_search_progress = False
            #solver.parameters.enumerate_all_solutions = True
            #status = solver.Solve(model, solution_counter)
            status = solver.Solve(model)
            if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
                #for idx, token in enumerate(uniq_tokens):
                #    token_vars = [token_idx_vars[i][idx] for i in range(page_data['tokenCount'])]
                #    print(f'{token}: {sum(solver.Value(v) for v in token_vars)}')
                # print sequence of tokens
                txt = ""
                for idx in range(page_data['tokenCount']):
                    for token_idx in range(uniq_token_count):
                        if solver.Value(token_idx_vars[idx][token_idx]):
                            if solver.Value(token_idx_start_vars[idx][token_idx]):
                                #txt += "\n"
                                pass
                            txt += uniq_tokens[token_idx] + " "
                            if solver.Value(token_idx_end_vars[idx][token_idx]):
                                txt += "\n"
                txt = re.sub(r' ([^a-zA-Z])', r'\1', txt)
                txt = txt.strip()
                embeddings.append(emb_model.encode(txt))
                progress.update(task, advance=1)
                console.print(txt)
                console.rule()

                if len(embeddings) % 10 == 0:
                    console.print("Saving embeddings...")
                    embeddings_arr = np.array(embeddings)
                    print(embeddings_arr.shape)
                    with open("embeddingsEXTRACTED.npy", "wb") as f:
                        np.save(f, embeddings_arr)

                    with open("embeddingsEXTRACTED.csv", "w") as f:
                        # create embedding of original
                        emb = emb_model.encode(original)
                        f.write("\t".join(str(x) for x in emb) + "\n")
                        for emb in embeddings:
                            f.write("\t".join(str(x) for x in emb) + "\n")
                        for emb in rand_embeddings:
                            f.write("\t".join(str(x) for x in emb) + "\n")

                    # create csv of ids
                    with open("ids.csv", "w") as f:
                        f.write("original\n")
                        for idx in range(len(embeddings)):
                            f.write(f"generated_{idx}\n")
                        for idx in range(len(rand_embeddings)):
                            f.write(f"random_{idx}\n")
                    console.rule()

