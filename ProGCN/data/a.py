with open('out.txt', 'r') as f, open('propara-results_dep-test.txt', 'w') as fo:
    for line in f:
        fo.write(line.replace('"', ''))
