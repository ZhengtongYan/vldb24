# Quantum-Inspired Digital Annealing for Join Ordering

This repository contains code and data artifacts for "Quantum-Inspired Digital Annealing for Join Ordering", submitted to VLDB 2024.

## Project Structure

FujitsuExperiments.py contains code for our experimental analysis, using Scripts/ProblemGenerator.py to load query data, where we consider (1) query graphs extracted from the standard SQLite benchmark [1], and (2) synthetic queries generated in accordance to the classic method by Steinbrunn et al. [2], using generation code by Trummer [3]. Data for all queries can be found in the ExperimentalAnalysis directory. 

Scripts/QUBOGenerator.py constructs JO-QUBO encodings based on our novel formulation method, allowing their deployment on the Fujitsu Digital Annealer [4]. Finally, Scripts/Postprocessing.py contains code for reading out join orders from each annealing solution obtained by the annealing device. Raw annealing results for all queries can be found in the ExperimentalAnalysis directory.

## References

[1] Richard D Hipp. 2020. SQLite. https://www.sqlite.org/index.htm

[2] Michael Steinbrunn, Guido Moerkotte, and Alfons Kemper. 1997. Heuristic and
randomized optimization for the join ordering problem. The VLDB journal 6
(1997), 191–208.

[3] Immanuel Trummer. 2016. Query Optimizer Library. https://github.com/itrummer/query-optimizer-lib

[4] Fujitsu Limited. 2023. Fujitsu Digital Annealer - solving large-scale combinatorial optimization problems instantly. https://www.fujitsu.com/emeia/services/business-services/digital-annealer/what-is-digital-annealer/
