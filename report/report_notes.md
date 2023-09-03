# Report Notes

## Base-Experiment (20 Minuten)
Base-Experiment: 0 temperature, basic prompting

### Result Interpretations

#### Best Prompts (in terms of metric performance)
- BloomZ
  - bertscore: 1,2,5,6,8,10,12,18
  - rouge: 1,2,5,6,8,10,12,18
  - coverage: 6,8,18 -> most "extractive"
- Falcon -> not that much of a difference here between the prompts
  - bertscore: 1,2,4
  - rouge: 2,5?
  - coverage:
- Llama2 -> not that much of a difference here between the prompts
  - bertscore: 2,5
  - rouge: 2
  - coverage: -> 1,5 are least extractive, all others overlapping (large)

Overall best prompts: 2,5
(2) Erstelle eine Zusammenfassung vom folgenden Artikel in 3 oder weniger Sätzen:\nArtikel: {article}\nZusammenfassung:\n
(5) {article} \n\nTL;DR:

#### Best prompts in terms of non-zero output
1,2,5

## Temperature Experiment
Temperature Experiment: 0.1, 0.5, 1
Selected Prompts: 1,2,5 + 3,4,6,7,12,18
