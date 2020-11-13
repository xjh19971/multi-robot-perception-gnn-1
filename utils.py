import os,datetime
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