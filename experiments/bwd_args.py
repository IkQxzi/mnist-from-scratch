
def inner_func(chain_dvals, true_classes):

    print(f'chain_dvals: {chain_dvals}')
    print(f'true_classes: {true_classes}')

    input_dvals = 'input_dvals1212'

    return input_dvals


def outer_func(*args, **kwargs):

    input_dvals = inner_func(*kwargs)
    


chain_dvals = 'chain1212'
true_classes = 'true1212'

outer_func(chain_dvals, true_classes)