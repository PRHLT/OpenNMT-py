def check_arguments(opt):

    # Check Max_Length
    max_length = 0
    with open(opt.tgt) as file:
        dataset = file.read().splitlines()
    for line in dataset:
        line = line.split()
        max_length = max(max_length, len(line))
    max_length += 20

    if opt.max_length < max_length:
        opt.max_length = max_length
        print('WARNING: MAX_LENGTH INCREASED TO {0}'.format(max_length))

    return opt