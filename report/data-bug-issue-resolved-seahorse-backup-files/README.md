Note:
- some pre-processing went from for the WikinewsTrunc3584
- I therefore re-pre-processed these bad articles, predicted new summaries, and predicted the SEAHORSE scores using the correct predictions
- I then manually replaced the relevant entries in the post-processed SEAHORSE results file
- ... by removing the bad entries
- ... and just appending the correct entries

SIDENOTE: I removed the code from the extend_results, because it is an extra case that never happens again
What I did: 
- I read in the original seahorse file as well as the bad seahorse post-processed file
- I calcualted a map from ground-truth summaries to original doc_ids
- I removed the entries with these bad doc_ids from the original seahorse file -> filtered file
- as backup I also saved the bad examples
- I then processed the examples that i just re-predicted into a file
- I then concatenated the good new predictions in the filtered file with the new predictions -> filled file
- I copied this file, renamed it to the original seahorse filename, and replaced it in the seahorse-results folder