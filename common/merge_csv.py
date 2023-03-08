import csv
import os
root_dir = '../log/Weifang_online/'
exp_file_pair = {
    "factor: 1.1": "trpo(eval).date(20190506, 20190526).confounder_False.eval_stochastic_True.budget_factor_1.1.evaluate_percent_20.0.ha_2/progress.csv",
    "factor: 1000": "trpo(eval).date(20190506, 20190526).confounder_False.eval_stochastic_True.budget_factor_True.evaluate_percent_1000.0.ha_2/progress.csv",
}

def merge_csv(exp_file_pair, root_dir, result_name):
    reader = csv.reader(open(root_dir + exp_file_pair[list(exp_file_pair.keys())[0]]))
    os.makedirs(os.path.dirname(result_name), exist_ok=True)
    headers = reader.__next__()
    headers.append('exp')
    writer = csv.DictWriter(open(result_name, 'w'), fieldnames=headers)
    writer.writeheader()
    for key, value in exp_file_pair.items():
        reader = csv.DictReader(open(root_dir + value))
        for row in reader:
            row['exp'] = key
            writer.writerow(row)

if __name__ =='__main__':
    root_dir = '../log/Weifang_online/'
    exp_file_pair = {
        "factor: 1.1": "trpo(eval).date(20190506, 20190526).confounder_False.eval_stochastic_True.budget_factor_1.1.evaluate_percent_1.0.ha_2/progress.csv",
        "factor: 1000": "trpo(eval).date(20190506, 20190526).confounder_False.eval_stochastic_True.budget_factor_1000.0.evaluate_percent_1.0.ha_2/progress.csv",
    }
    result_name = root_dir + 'results/budget_test.csv'
    merge_csv(exp_file_pair, root_dir, result_name)