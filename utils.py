import os
from datetime import datetime
import re
import pandas as pd

pattern = re.compile(
    r'.*step (\d+) \|  train loss \[depth: (\S+)\] valid loss \[depth: (\S+)\]')
df_header = (
    'Step',
    'TrDep',
    'TeDep',
)

# Logging function
def log(fname, s):
    if not os.path.isdir(os.path.dirname(fname)):
        os.system(f'mkdir -p {os.path.dirname(fname)}')
    f = open(fname, 'a')
    try:
        f.write(f'{str(datetime.now())}: {s}\n')
    except Exception:
        f.write('Print error\n')
    f.close()


def format_losses(loss, split='train'):
    log_string = ' '
    log_string += f'{split} loss ['
    log_string += f'depth: {loss:.5f}'
    log_string += ']'
    return log_string

def load_log_file(file_name):
    with open(file_name, 'r') as log_file:
        log_list = list()
        for n, line in enumerate(log_file):
            match = re.match(pattern, line)
            if match is not None:
                log_list.append(tuple(
                    int(g) if i is 0 else float(g) for i, g in enumerate(match.groups())
                ))
    # Create data frame
    df = pd.DataFrame(data=log_list, columns=df_header)
    return df
