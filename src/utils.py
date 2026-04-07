import os

PROPER_ORDER = ['SuperFuture', 'Apples', 'WorldNow', 'Electronics123', 'Photons', 
                'SpaceNow', 'PearPear', 'PositiveCorrelation', 'BetterTechnology', 'ABCDE',
                'EnviroLike', 'Moneymakers', 'Fuel4', 'MarsProject', 'CPU-XYZ', 
                'RoboticsX', 'Lasers', 'WaterForce', 'SafeAndCare', 'BetterTomorrow']
ALPHABETICAL_ORDER = sorted(PROPER_ORDER)


def read_data(filename):
    with open(filename, 'r') as f:
        data = f.read().strip().splitlines()
    name = data[0]
    time_length = int(data[1])
    prices = [float(x[1]) for x in (line.split() for line in data[2:])]
    return name, prices

def read_all_files(DIR_path):
    stocks = {}
    for filename in os.listdir(DIR_path):
        if filename.endswith('.txt'):
            name, prices = read_data(os.path.join(DIR_path, filename))
            stocks[name] = prices
    return stocks

def write_solutions(solution, file_name):
    my_order = ALPHABETICAL_ORDER
    assert set(my_order) == set(PROPER_ORDER)

    risk, gain, portfolio = solution

    results_map = dict(zip(my_order, portfolio))
    ordered_portfolio = [str(results_map[name]) for name in PROPER_ORDER]

    with open(file_name, 'w') as f:
        f.write(f'{gain} ')
        f.write(f'{risk} ')
        f.write(' '.join([str(i) for i in ordered_portfolio]))

def read_solutions(filename):
    with open(filename, 'r') as f:
        data = f.read().split()
    est_return = float(data[0])
    est_risk = float(data[1])
    weights = list(map(float, data[2:]))
    return weights