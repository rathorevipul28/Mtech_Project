import evalQA

paraIdFile = 'para_id.test.txt'
goldFile = 'test.gold.tsv'
predictionFile = 'query_entity/results1/propara-results-test.txt'
allMoveFile = 'all-moves.full-grid.tsv'
print(evalQA.main(paraIdFile, goldFile, predictionFile, allMoveFile))

