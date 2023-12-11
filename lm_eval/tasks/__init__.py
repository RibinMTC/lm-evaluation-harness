import os
from pprint import pprint
from typing import List, Union, Dict

import sacrebleu
import lm_eval.base
import json

from . import superglue
from . import glue
from . import arc
from . import coqa
from . import race
from . import webqs
from . import anli
from . import wsc273
from . import winogrande
from . import quac
from . import hellaswag
from . import swag
from . import openbookqa
from . import squad
from . import naturalqs
from . import sat
from . import arithmetic
from . import lambada
from . import piqa
from . import prost
from . import mc_taco
from . import triviaqa
from . import pubmedqa
from . import sciq
from . import qasper
from . import qa4mre
from . import translation
from . import headqa
from . import mathqa
from . import hendrycks_ethics
from . import drop
from . import unscramble
from . import logiqa
from . import hendrycks_test
from . import hendrycks_math
from . import cbt
from . import lambada_cloze
from . import pile
from . import wikitext
from . import lambada_multilingual
from . import mutual
from . import truthfulqa
from . import blimp
from . import asdiv
from . import gsm8k
from . import storycloze
from . import toxigen
from . import crowspairs
from . import json
from . import xcopa
from . import bigbench
from . import xstorycloze
from . import xwinograd
from . import pawsx
from . import xnli
from . import mgsm
from . import germanquad
from . import germeval2017
from . import x_stance
from . import cnndm_paraphrase
from . import factcc_hallucination_classification
from . import frank_hallucination_classification
from . import xsum_faith_hallucination_classification
from . import swisstext23_summarization
from . import llm_summarization_mt
from . import seahorse_classification
from . import seahorse_classification_automatic

########################################
# Translation tasks
########################################

# 6 total
gpt3_translation_benchmarks = {
    "wmt14": ["en-fr", "fr-en"],  # French
    "wmt16": ["en-ro", "ro-en", "de-en", "en-de"],  # German, Romanian
}

# 28 total
selected_translation_benchmarks = {
    **gpt3_translation_benchmarks,
    "wmt20": sacrebleu.get_langpairs_for_testset("wmt20"),
    "iwslt17": ["en-ar", "ar-en"],  # Arabic
}

# 319 total
all_translation_benchmarks = {
    ts: sacrebleu.get_langpairs_for_testset(ts)
    for ts in sacrebleu.get_available_testsets()
}

########################################
# All tasks
########################################


TASK_REGISTRY = {
    # GLUE
    "cola": glue.CoLA,
    "mnli": glue.MNLI,
    "mnli_mismatched": glue.MNLIMismatched,
    "mrpc": glue.MRPC,
    "rte": glue.RTE,
    "qnli": glue.QNLI,
    "qqp": glue.QQP,
    # "stsb": glue.STSB, # not implemented yet
    "sst": glue.SST,
    "wnli": glue.WNLI,
    # SuperGLUE
    "boolq": superglue.BoolQ,
    "cb": superglue.CommitmentBank,
    "copa": superglue.Copa,
    "multirc": superglue.MultiRC,
    "record": superglue.ReCoRD,
    "wic": superglue.WordsInContext,
    "wsc": superglue.SGWinogradSchemaChallenge,
    # Order by benchmark/genre?
    "coqa": coqa.CoQA,
    "drop": drop.DROP,
    "lambada_openai": lambada.LambadaOpenAI,
    "lambada_standard": lambada.LambadaStandard,
    "lambada_openai_cloze": lambada_cloze.LambadaOpenAICloze,
    "lambada_standard_cloze": lambada_cloze.LambadaStandardCloze,
    # multilingual lambada
    **lambada_multilingual.construct_tasks(),
    "wikitext": wikitext.WikiText,
    # "cbt-cn": cbt.CBTCN, # disabled pending context length fix
    # "cbt-ne": cbt.CBTNE, # disabled pending context length fix
    "piqa": piqa.PiQA,
    "prost": prost.PROST,
    "mc_taco": mc_taco.MCTACO,
    # Science related
    "pubmedqa": pubmedqa.Pubmed_QA,
    "sciq": sciq.SciQ,
    "qasper": qasper.QASPER,
    "qa4mre_2011": qa4mre.QA4MRE_2011,
    "qa4mre_2012": qa4mre.QA4MRE_2012,
    "qa4mre_2013": qa4mre.QA4MRE_2013,
    "triviaqa": triviaqa.TriviaQA,
    "arc_easy": arc.ARCEasy,
    "arc_challenge": arc.ARCChallenge,
    # "quac": quac.QuAC, # not implemented yet
    "logiqa": logiqa.LogiQA,
    "hellaswag": hellaswag.HellaSwag,
    "swag": swag.SWAG,
    "openbookqa": openbookqa.OpenBookQA,
    "squad2": squad.SQuAD2,
    "race": race.RACE,
    # "naturalqs": naturalqs.NaturalQs, # not implemented yet
    "headqa": headqa.HeadQAEsDeprecated,  # for backwards compat - headqa used to default to es
    "headqa_es": headqa.HeadQAEs,
    "headqa_en": headqa.HeadQAEn,
    "mathqa": mathqa.MathQA,
    "webqs": webqs.WebQs,
    "wsc273": wsc273.WinogradSchemaChallenge273,
    "winogrande": winogrande.Winogrande,
    "anli_r1": anli.ANLIRound1,
    "anli_r2": anli.ANLIRound2,
    "anli_r3": anli.ANLIRound3,
    "ethics_cm": hendrycks_ethics.EthicsCM,
    "ethics_deontology": hendrycks_ethics.EthicsDeontology,
    "ethics_justice": hendrycks_ethics.EthicsJustice,
    "ethics_utilitarianism_original": hendrycks_ethics.EthicsUtilitarianismOriginal,
    "ethics_utilitarianism": hendrycks_ethics.EthicsUtilitarianism,
    "ethics_virtue": hendrycks_ethics.EthicsVirtue,
    "truthfulqa_mc": truthfulqa.TruthfulQAMultipleChoice,
    "truthfulqa_gen": truthfulqa.TruthfulQAGeneration,
    # dialogue
    "mutual": mutual.MuTual,
    "mutual_plus": mutual.MuTualPlus,
    # math
    "math_algebra": hendrycks_math.MathAlgebra,
    "math_counting_and_prob": hendrycks_math.MathCountingAndProbability,
    "math_geometry": hendrycks_math.MathGeometry,
    "math_intermediate_algebra": hendrycks_math.MathIntermediateAlgebra,
    "math_num_theory": hendrycks_math.MathNumberTheory,
    "math_prealgebra": hendrycks_math.MathPrealgebra,
    "math_precalc": hendrycks_math.MathPrecalculus,
    "math_asdiv": asdiv.Asdiv,
    "gsm8k": gsm8k.GradeSchoolMath8K,
    # arithmetic
    "arithmetic_2da": arithmetic.Arithmetic2DPlus,
    "arithmetic_2ds": arithmetic.Arithmetic2DMinus,
    "arithmetic_3da": arithmetic.Arithmetic3DPlus,
    "arithmetic_3ds": arithmetic.Arithmetic3DMinus,
    "arithmetic_4da": arithmetic.Arithmetic4DPlus,
    "arithmetic_4ds": arithmetic.Arithmetic4DMinus,
    "arithmetic_5da": arithmetic.Arithmetic5DPlus,
    "arithmetic_5ds": arithmetic.Arithmetic5DMinus,
    "arithmetic_2dm": arithmetic.Arithmetic2DMultiplication,
    "arithmetic_1dc": arithmetic.Arithmetic1DComposite,
    # TODO Perhaps make these groups of tasks
    #   e.g. anli, arithmetic, openai_translations, harness_translations
    # hendrycksTest (57 tasks)
    **hendrycks_test.create_all_tasks(),
    # e.g. wmt14-fr-en
    **translation.create_tasks_from_benchmarks(gpt3_translation_benchmarks),
    # chef's selection, mostly wmt20
    **translation.create_tasks_from_benchmarks(selected_translation_benchmarks),
    # Word Scrambling and Manipulation Tasks
    "anagrams1": unscramble.Anagrams1,
    "anagrams2": unscramble.Anagrams2,
    "cycle_letters": unscramble.CycleLetters,
    "random_insertion": unscramble.RandomInsertion,
    "reversed_words": unscramble.ReversedWords,
    # Pile
    "pile_arxiv": pile.PileArxiv,
    "pile_books3": pile.PileBooks3,
    "pile_bookcorpus2": pile.PileBookCorpus2,
    "pile_dm-mathematics": pile.PileDmMathematics,
    "pile_enron": pile.PileEnron,
    "pile_europarl": pile.PileEuroparl,
    "pile_freelaw": pile.PileFreeLaw,
    "pile_github": pile.PileGithub,
    "pile_gutenberg": pile.PileGutenberg,
    "pile_hackernews": pile.PileHackernews,
    "pile_nih-exporter": pile.PileNIHExporter,
    "pile_opensubtitles": pile.PileOpenSubtitles,
    "pile_openwebtext2": pile.PileOpenWebText2,
    "pile_philpapers": pile.PilePhilPapers,
    "pile_pile-cc": pile.PilePileCc,
    "pile_pubmed-abstracts": pile.PilePubmedAbstracts,
    "pile_pubmed-central": pile.PilePubmedCentral,
    "pile_stackexchange": pile.PileStackExchange,
    "pile_uspto": pile.PileUspto,
    "pile_ubuntu-irc": pile.PileUbuntuIrc,
    "pile_wikipedia": pile.PileWikipedia,
    "pile_youtubesubtitles": pile.PileYoutubeSubtitles,
    # BLiMP
    "blimp_adjunct_island": blimp.BlimpAdjunctIsland,
    "blimp_anaphor_gender_agreement": blimp.BlimpAnaphorGenderAgreement,
    "blimp_anaphor_number_agreement": blimp.BlimpAnaphorNumberAgreement,
    "blimp_animate_subject_passive": blimp.BlimpAnimateSubjectPassive,
    "blimp_animate_subject_trans": blimp.BlimpAnimateSubjectTrans,
    "blimp_causative": blimp.BlimpCausative,
    "blimp_complex_NP_island": blimp.BlimpComplex_NPIsland,
    "blimp_coordinate_structure_constraint_complex_left_branch": blimp.BlimpCoordinateStructureConstraintComplexLeftBranch,
    "blimp_coordinate_structure_constraint_object_extraction": blimp.BlimpCoordinateStructureConstraintObjectExtraction,
    "blimp_determiner_noun_agreement_1": blimp.BlimpDeterminerNounAgreement_1,
    "blimp_determiner_noun_agreement_2": blimp.BlimpDeterminerNounAgreement_2,
    "blimp_determiner_noun_agreement_irregular_1": blimp.BlimpDeterminerNounAgreementIrregular_1,
    "blimp_determiner_noun_agreement_irregular_2": blimp.BlimpDeterminerNounAgreementIrregular_2,
    "blimp_determiner_noun_agreement_with_adj_2": blimp.BlimpDeterminerNounAgreementWithAdj_2,
    "blimp_determiner_noun_agreement_with_adj_irregular_1": blimp.BlimpDeterminerNounAgreementWithAdjIrregular_1,
    "blimp_determiner_noun_agreement_with_adj_irregular_2": blimp.BlimpDeterminerNounAgreementWithAdjIrregular_2,
    "blimp_determiner_noun_agreement_with_adjective_1": blimp.BlimpDeterminerNounAgreementWithAdjective_1,
    "blimp_distractor_agreement_relational_noun": blimp.BlimpDistractorAgreementRelationalNoun,
    "blimp_distractor_agreement_relative_clause": blimp.BlimpDistractorAgreementRelativeClause,
    "blimp_drop_argument": blimp.BlimpDropArgument,
    "blimp_ellipsis_n_bar_1": blimp.BlimpEllipsisNBar_1,
    "blimp_ellipsis_n_bar_2": blimp.BlimpEllipsisNBar_2,
    "blimp_existential_there_object_raising": blimp.BlimpExistentialThereObjectRaising,
    "blimp_existential_there_quantifiers_1": blimp.BlimpExistentialThereQuantifiers_1,
    "blimp_existential_there_quantifiers_2": blimp.BlimpExistentialThereQuantifiers_2,
    "blimp_existential_there_subject_raising": blimp.BlimpExistentialThereSubjectRaising,
    "blimp_expletive_it_object_raising": blimp.BlimpExpletiveItObjectRaising,
    "blimp_inchoative": blimp.BlimpInchoative,
    "blimp_intransitive": blimp.BlimpIntransitive,
    "blimp_irregular_past_participle_adjectives": blimp.BlimpIrregularPastParticipleAdjectives,
    "blimp_irregular_past_participle_verbs": blimp.BlimpIrregularPastParticipleVerbs,
    "blimp_irregular_plural_subject_verb_agreement_1": blimp.BlimpIrregularPluralSubjectVerbAgreement_1,
    "blimp_irregular_plural_subject_verb_agreement_2": blimp.BlimpIrregularPluralSubjectVerbAgreement_2,
    "blimp_left_branch_island_echo_question": blimp.BlimpLeftBranchIslandEchoQuestion,
    "blimp_left_branch_island_simple_question": blimp.BlimpLeftBranchIslandSimpleQuestion,
    "blimp_matrix_question_npi_licensor_present": blimp.BlimpMatrixQuestionNpiLicensorPresent,
    "blimp_npi_present_1": blimp.BlimpNpiPresent_1,
    "blimp_npi_present_2": blimp.BlimpNpiPresent_2,
    "blimp_only_npi_licensor_present": blimp.BlimpOnlyNpiLicensorPresent,
    "blimp_only_npi_scope": blimp.BlimpOnlyNpiScope,
    "blimp_passive_1": blimp.BlimpPassive_1,
    "blimp_passive_2": blimp.BlimpPassive_2,
    "blimp_principle_A_c_command": blimp.BlimpPrinciple_ACCommand,
    "blimp_principle_A_case_1": blimp.BlimpPrinciple_ACase_1,
    "blimp_principle_A_case_2": blimp.BlimpPrinciple_ACase_2,
    "blimp_principle_A_domain_1": blimp.BlimpPrinciple_ADomain_1,
    "blimp_principle_A_domain_2": blimp.BlimpPrinciple_ADomain_2,
    "blimp_principle_A_domain_3": blimp.BlimpPrinciple_ADomain_3,
    "blimp_principle_A_reconstruction": blimp.BlimpPrinciple_AReconstruction,
    "blimp_regular_plural_subject_verb_agreement_1": blimp.BlimpRegularPluralSubjectVerbAgreement_1,
    "blimp_regular_plural_subject_verb_agreement_2": blimp.BlimpRegularPluralSubjectVerbAgreement_2,
    "blimp_sentential_negation_npi_licensor_present": blimp.BlimpSententialNegationNpiLicensorPresent,
    "blimp_sentential_negation_npi_scope": blimp.BlimpSententialNegationNpiScope,
    "blimp_sentential_subject_island": blimp.BlimpSententialSubjectIsland,
    "blimp_superlative_quantifiers_1": blimp.BlimpSuperlativeQuantifiers_1,
    "blimp_superlative_quantifiers_2": blimp.BlimpSuperlativeQuantifiers_2,
    "blimp_tough_vs_raising_1": blimp.BlimpToughVsRaising_1,
    "blimp_tough_vs_raising_2": blimp.BlimpToughVsRaising_2,
    "blimp_transitive": blimp.BlimpTransitive,
    "blimp_wh_island": blimp.BlimpWhIsland,
    "blimp_wh_questions_object_gap": blimp.BlimpWhQuestionsObjectGap,
    "blimp_wh_questions_subject_gap": blimp.BlimpWhQuestionsSubjectGap,
    "blimp_wh_questions_subject_gap_long_distance": blimp.BlimpWhQuestionsSubjectGapLongDistance,
    "blimp_wh_vs_that_no_gap": blimp.BlimpWhVsThatNoGap,
    "blimp_wh_vs_that_no_gap_long_distance": blimp.BlimpWhVsThatNoGapLongDistance,
    "blimp_wh_vs_that_with_gap": blimp.BlimpWhVsThatWithGap,
    "blimp_wh_vs_that_with_gap_long_distance": blimp.BlimpWhVsThatWithGapLongDistance,
    "toxigen": toxigen.ToxiGen,
    "crows_pairs_english": crowspairs.CrowsPairsEnglish,
    "crows_pairs_english_race_color": crowspairs.CrowsPairsEnglishRaceColor,
    "crows_pairs_english_socioeconomic": crowspairs.CrowsPairsEnglishSocioeconomic,
    "crows_pairs_english_gender": crowspairs.CrowsPairsEnglishGender,
    "crows_pairs_english_age": crowspairs.CrowsPairsEnglishAge,
    "crows_pairs_english_religion": crowspairs.CrowsPairsEnglishReligion,
    "crows_pairs_english_disability": crowspairs.CrowsPairsEnglishDisability,
    "crows_pairs_english_sexual_orientation": crowspairs.CrowsPairsEnglishSexualOrientation,
    "crows_pairs_english_nationality": crowspairs.CrowsPairsEnglishNationality,
    "crows_pairs_english_physical_appearance": crowspairs.CrowsPairsEnglishPhysicalAppearance,
    "crows_pairs_english_autre": crowspairs.CrowsPairsEnglishAutre,
    "crows_pairs_french": crowspairs.CrowsPairsFrench,
    "crows_pairs_french_race_color": crowspairs.CrowsPairsFrenchRaceColor,
    "crows_pairs_french_socioeconomic": crowspairs.CrowsPairsFrenchSocioeconomic,
    "crows_pairs_french_gender": crowspairs.CrowsPairsFrenchGender,
    "crows_pairs_french_age": crowspairs.CrowsPairsFrenchAge,
    "crows_pairs_french_religion": crowspairs.CrowsPairsFrenchReligion,
    "crows_pairs_french_disability": crowspairs.CrowsPairsFrenchDisability,
    "crows_pairs_french_sexual_orientation": crowspairs.CrowsPairsFrenchSexualOrientation,
    "crows_pairs_french_nationality": crowspairs.CrowsPairsFrenchNationality,
    "crows_pairs_french_physical_appearance": crowspairs.CrowsPairsFrenchPhysicalAppearance,
    "crows_pairs_french_autre": crowspairs.CrowsPairsFrenchAutre,
    "germanquad_open_qa": germanquad.GermanQuadOpenDomainQATask,
    "germeval2017": germeval2017.GermEval2017,
    "x_stance_de": x_stance.XStanceDE,
    "cnn_dm_paraphrase": cnndm_paraphrase.CnnDMParaphraseTask,
    "factcc_hallucination_classification": factcc_hallucination_classification.FactCCHallucinationClassificationTask,
    "frank_hallucination_classification": frank_hallucination_classification.FrankHallucinationClassificationTask,
    "xsum_faith_hallucination_classification": xsum_faith_hallucination_classification.XsumFaithHallucinationClassificationTask,
    "swisstext23_summarization": swisstext23_summarization.SwissText23SummarizationTask,
    "SummarizationLocal": llm_summarization_mt.SummarizationTaskLocal,
    "RepeatExperimentBugfix_0_Llama7b_100_8b": llm_summarization_mt.RepeatExperiments_after_bugfix_0_Llama7b,
    "RepeatExperimentBugfix_0_Llama70b_100_8b": llm_summarization_mt.RepeatExperiments_after_bugfix_0_Llama70b,
    "RepeatExperimentBugfix_1_Llama70b_100_8b": llm_summarization_mt.RepeatExperiments_after_bugfix_1_Llama70b,
    "RepeatExperimentBugfix_2_Llama70b_100_8b": llm_summarization_mt.RepeatExperiments_after_bugfix_2_Llama70b,
    "RepeatExperimentBugfix_3_Llama70b_100_8b": llm_summarization_mt.RepeatExperiments_after_bugfix_3_Llama70b,
    "RepeatExperimentBugfix_4_Llama70b_100_8b": llm_summarization_mt.RepeatExperiments_after_bugfix_4_Llama70b,
    "SummSampleSmol_20Minuten_1": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSampleSmol_20Minuten_2": llm_summarization_mt.SummSampleSmol_20Minuten,
    # "SummLtM1_20Minuten_21": llm_summarization_mt.SummarizationTask_20Minuten,
    # "SummLtM1_20Minuten_22": llm_summarization_mt.SummarizationTask_20Minuten,
    # "SummLtM1_20Minuten_30": llm_summarization_mt.SummarizationTask_20Minuten,
    # "SummLtM1_20Minuten_31": llm_summarization_mt.SummarizationTask_20Minuten,
    # "SummLtM1_20Minuten_32": llm_summarization_mt.SummarizationTask_20Minuten,
    # "SummLtM1_20Minuten_33": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummLtM1_20Minuten_21_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummLtM1_20Minuten_22_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummLtM1_20Minuten_30_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummLtM1_20Minuten_31_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummLtM1_20Minuten_32_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummLtM1_20Minuten_33_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    # "SummLtM2_20min21_23_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    # "SummLtM2_20min22_24_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    # "SummLtM2_20min30_34_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    # "SummLtM2_20min32_34_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    # "SummLtM2_20min31_35_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    # "SummLtM2_20min33_35_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummLtM1_20min0_21_8b": llm_summarization_mt.SummShard0_20Minuten,
    "SummLtM1_20min0_22_8b": llm_summarization_mt.SummShard0_20Minuten,
    "SummLtM1_20min0_30_8b": llm_summarization_mt.SummShard0_20Minuten,
    "SummLtM1_20min0_31_8b": llm_summarization_mt.SummShard0_20Minuten,
    "SummLtM1_20min0_32_8b": llm_summarization_mt.SummShard0_20Minuten,
    "SummLtM1_20min0_33_8b": llm_summarization_mt.SummShard0_20Minuten,
    "SummLtm2_20minLtm2p22S_35_8b": llm_summarization_mt.SummLtM2_20min_prompt22_start,
    "SummLtm2_20minLtm2p22S_37_8b": llm_summarization_mt.SummLtM2_20min_prompt22_start,
    "SummLtm2_20minLtm2p22E_35_8b": llm_summarization_mt.SummLtM2_20min_prompt22_end,
    "SummLtm2_20minLtm2p22E_37_8b": llm_summarization_mt.SummLtM2_20min_prompt22_end,
    "SummLtm2_20minLtm2p31S_35_8b": llm_summarization_mt.SummLtM2_20min_prompt31_start,
    "SummLtm2_20minLtm2p31S_37_8b": llm_summarization_mt.SummLtM2_20min_prompt31_start,
    "SummLtm2_20minLtm2p31E_35_8b": llm_summarization_mt.SummLtM2_20min_prompt31_end,
    "SummLtm2_20minLtm2p31E_37_8b": llm_summarization_mt.SummLtM2_20min_prompt31_end,
    "SummLtm2_20minLtm2p33S_35_8b": llm_summarization_mt.SummLtM2_20min_prompt33_start,
    "SummLtm2_20minLtm2p33S_37_8b": llm_summarization_mt.SummLtM2_20min_prompt33_start,
    "SummLtm2_20minLtm2p33E_35_8b": llm_summarization_mt.SummLtM2_20min_prompt33_end,
    "SummLtm2_20minLtm2p33E_37_8b": llm_summarization_mt.SummLtM2_20min_prompt33_end,
    # "SummLtM2_20min021_23_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    # "SummLtM2_20min022_24_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    # "SummLtM2_20min030_34_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    # "SummLtM2_20min032_34_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    # "SummLtM2_20min031_35_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    # "SummLtM2_20min033_35_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "MDSChain_WikinewsS2_100_8b": llm_summarization_mt.MDS_CHAIN_Wikinews_Stage2_SDS_Prep,
    "MDSChain_WikinewsClustS2_100_8b": llm_summarization_mt.MDS_CHAIN_Wikinews_Clust_Stage2_SDS_Prep,
    "MDSChain_WikinewsClustDistS2_100_8b": llm_summarization_mt.MDS_CHAIN_Wikinews_ClustDist_Stage2_SDS_Prep,
    "MDS_WikinewsTrunc3584_52_8b": llm_summarization_mt.MDS_Baseline_truncated_3584_wikinewssum,
    "MDSFCO_WikiCD040SSimDyn1024_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clustDist_04_0shot_simSize_dyn_chunk_1024,
    "MDSFCO_WikiCD040SSimDyn1536_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clustDist_04_0shot_simSize_dyn_chunk_1536,
    "MDSFCO_WikiCD041SSimDyn1024_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clustDist_04_1shot_seed42_simSize_dyn_chunk_1024,
    "MDSFCO_WikiCD042SSimDyn1024_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clustDist_04_2shot_seed42_simSize_dyn_chunk_1024,
    "MDSFCO_WikiCD043SSimDyn1024_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clustDist_04_3shot_seed42_simSize_dyn_chunk_1024,
    "MDSFCO_WikiCD041SSimDyn1536_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clustDist_04_1shot_seed42_simSize_dyn_chunk_1536,
    "MDSFCO_WikiCD042SSimDyn1536_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clustDist_04_2shot_seed42_simSize_dyn_chunk_1536,
    "MDSFCO_WikiCD043SSimDyn1536_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clustDist_04_3shot_seed42_simSize_dyn_chunk_1536,
    "MDSFCO_WikiCD050SSimDyn1536_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clustDist_05_0shot_simSize_dyn_chunk_1536,
    "MDSFCO_WikiCD060SSimDyn1536_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clustDist_06_0shot_simSize_dyn_chunk_1536,
    "MDSFCO_WikiCD051SSimDyn1536_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clustDist_05_1shot_seed42_simSize_dyn_chunk_1536,
    "MDSFCO_WikiCD061SSimDyn1536_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clustDist_06_1shot_seed42_simSize_dyn_chunk_1536,
    "MDSFCO_WikiCD052SSimDyn1536_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clustDist_05_2shot_seed42_simSize_dyn_chunk_1536,
    "MDSFCO_WikiCD062SSimDyn1536_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clustDist_06_2shot_seed42_simSize_dyn_chunk_1536,
    "MDSFCO_WikiCD053SSimDyn1536_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clustDist_05_3shot_seed42_simSize_dyn_chunk_1536,
    "MDSFCO_WikiCD063SSimDyn1536_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clustDist_06_3shot_seed42_simSize_dyn_chunk_1536,
    "MDSFCO_WikiCl0SSimDyn1536_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clust_0shot_simSize_dyn_chunk_1536,
    "MDSFCO_WikiCl1SSimDyn1536_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clust_1shot_seed42_simSize_dyn_chunk_1536,
    "MDSFCO_WikiCl2SSimDyn1536_100_8b": llm_summarization_mt.MDS_FCO_wikinews_clust_2shot_seed42_simSize_dyn_chunk_1536,
    "MDS_MultinewsTrunc3584_52_8b": llm_summarization_mt.MDS_Baseline_truncated_3584_multinews,
    "MDSFCO_MultiCD040SSimDyn1024_100_8b": llm_summarization_mt.MDS_FCO_multinews_clustDist_04_0shot_simSize_dyn_chunk_1024,
    "MDSFCO_MultiCD040SSimDyn1536_100_8b": llm_summarization_mt.MDS_FCO_multinews_clustDist_04_0shot_simSize_dyn_chunk_1536,
    "MDSFCO_WikiCh1024_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Cheat_1024,
    "MDSFCO_WikiCh1536_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Cheat_1536,
    "MDSFCO_WikiLe1024_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Lead_1024,
    "MDSFCO_WikiLe1536_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Lead_1536,
    "MDSFCO_WikiLe1S21024_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Lead_1shot_20min_42_1024,
    "MDSFCO_WikiLe1S21536_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Lead_1shot_20min_42_1536,
    "MDSFCO_WikiRa1024_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Rand_1024,
    "MDSFCO_WikiRa1536_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Rand_1536,
    "MDSFCO_WikiRa1S21024_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Rand_1shot_20min_42_1024,
    "MDSFCO_WikiRa1S21536_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Rand_1shot_20min_42_1536,
    "MDSFCO_WikiRa1SW1024_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Rand_1shot_Wikinews_42_1024,
    "MDSFCO_WikiRa1SW1536_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Rand_1shot_Wikinews_42_1536,
    "MDSFCO_WikiCl0N1024_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Clust_0shot_1024,
    "MDSFCO_WikiCl0N1536_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Clust_0shot_1536,
    "MDSFCO_WikiCl0N2048_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Clust_0shot_2048,
    "MDSFCO_WikiCl1N21024_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Clust_1shot_20min_42_1024,
    "MDSFCO_WikiCl1N21536_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Clust_1shot_20min_42_1536,
    "MDSFCO_WikiCl1N22048_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Clust_1shot_20min_42_2048,
    "MDSFCO_WikiCl2S21024_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Clust_2shot_20min_42_1024,
    "MDSFCO_WikiCl2S21536_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Clust_2shot_20min_42_1536,
    "MDSFCO_WikiCl2S22048_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Clust_2shot_20min_42_2048,
    "MDSFCO_WikiCl1SW1024_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Clust_1shot_Wikinews_42_1024,
    "MDSFCO_WikiCl1SW1536_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Clust_1shot_Wikinews_42_1536,
    "MDSFCO_WikiCl2SW1024_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Clust_2shot_Wikinews_42_1024,
    "MDSFCO_WikiCl2SW1536_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_Clust_2shot_Wikinews_42_1536,
    "MDSFCO_WikiDi0S1024_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_DistMMR_0shot_1024,
    "MDSFCO_WikiDi0S1536_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_DistMMR_0shot_1536,
    "MDSFCO_WikiDi1S21024_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_DistMMR_1shot_20min_42_1024,
    "MDSFCO_WikiDi1S21536_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_DistMMR_1shot_20min_42_1536,
    "MDSFCO_WikiDi2S21024_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_DistMMR_2shot_20min_42_1024,
    "MDSFCO_WikiDi2S21536_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_DistMMR_2shot_20min_42_1536,
    "MDSFCO_WikiDi1SW1024_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_DistMMR_1shot_Wikinews_42_1024,
    "MDSFCO_WikiDi1SW1536_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_DistMMR_1shot_Wikinews_42_1536,
    "MDSFCO_WikiDi2SW1024_100_8b": llm_summarization_mt.MDS_FCO_Wikinews_DistMMR_2shot_Wikinews_42_1024,
    "MDSSumm_Wikinews_50": llm_summarization_mt.SummarizationTask_Wikinewssum,
    "MDSSumm_Wikinews_51": llm_summarization_mt.SummarizationTask_Wikinewssum,
    "MDSSumm_Wikinews_52": llm_summarization_mt.SummarizationTask_Wikinewssum,
    "MDSSumm_Wikinews_50_8b": llm_summarization_mt.SummarizationTask_Wikinewssum,
    "MDSSumm_Wikinews_51_8b": llm_summarization_mt.SummarizationTask_Wikinewssum,
    "MDSSumm_Wikinews_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum,
    "MDSSumm_WikinewsClean_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_Cleaned,
    "MDSSumm_WikinewsSL_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_SingleLine,
    "MDSSumm_WikinewsSLR_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_SingleLine_Shuffled,
    "MDSSumm_WikinewsSimple_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_Simple,
    "MDSSumm_WikinewsSimpleS_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_Simple_Shuffled,
    "MDSSumm_WikinewsSimpleA_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_Simple_ArticleIdxAnn,
    "MDSSumm_WikinewsSimpleAS_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_Simple_ArticleIdxAnn_Shuffled,
    "MDSSumm_WikinewsSC32_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_Simple_Chunked_32,
    "MDSSumm_WikinewsSC64_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_Simple_Chunked_64,
    "MDSSumm_WikinewsSC128_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_Simple_Chunked_128,
    "MDSSumm_WikinewsSC256_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_Simple_Chunked_256,
    "MDSSumm_WikinewsSC512_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_Simple_Chunked_512,
    "MDSSumm_WikinewsSCS2_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_Simple_Chunked_Sentences_2,
    "MDSSumm_WikinewsSCS4_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_Simple_Chunked_Sentences_4,
    "MDSSumm_WikinewsSCS8_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_Simple_Chunked_Sentences_8,
    "MDSSumm_WikinewsSCS16_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_Simple_Chunked_Sentences_16,
    "MDSSumm_WikinewsSCS32_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_Simple_Chunked_Sentences_32,
    "MDS2S_WikinewsSplit_2_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_2Stage_Split_Input_Docs,
    "MDS2S_WikinewsSplit_40_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_2Stage_Split_Input_Docs,
    "MDS2S_WikinewsSplit_41_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_2Stage_Split_Input_Docs,
    "MDS2S_WikinewsSplitS2OP41_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_2Stage_Split_Input_Docs_Stage2_BasePrompt41_OriginalOrder,
    "MDS2S_WikinewsSplitS2SP41_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_2Stage_Split_Input_Docs_Stage2_BasePrompt41_Shuffled,
    "MDS2S_WikinewsSplitS2O_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_2Stage_Split_Input_Docs_Stage2_OriginalOrder,
    "MDS2S_WikinewsSplitS2S_52_8b": llm_summarization_mt.SummarizationTask_Wikinewssum_2Stage_Split_Input_Docs_Stage2_Shuffled,
    "MDS_WikinewsClust1R_42_8b": llm_summarization_mt.MDS_WikinewsSum_ClusterChunk_1_Random,
    "MDS_WikinewsClust1O_42_8b": llm_summarization_mt.MDS_WikinewsSum_ClusterChunk_1_Original_Order,
    "MDS_WikinewsClust1C_42_8b": llm_summarization_mt.MDS_WikinewsSum_ClusterChunk_1_Cluster_Size,
    "MDS_WikinewsClust3R_42_8b": llm_summarization_mt.MDS_WikinewsSum_ClusterChunk_3_Random,
    "MDS_WikinewsClust3O_42_8b": llm_summarization_mt.MDS_WikinewsSum_ClusterChunk_3_Original_Order,
    "MDS_WikinewsClust3C_42_8b": llm_summarization_mt.MDS_WikinewsSum_ClusterChunk_3_Cluster_Size,
    "MDS_WikinewsClust5R_42_8b": llm_summarization_mt.MDS_WikinewsSum_ClusterChunk_5_Random,
    "MDS_WikinewsClust5O_42_8b": llm_summarization_mt.MDS_WikinewsSum_ClusterChunk_5_Original_Order,
    "MDS_WikinewsClust5C_42_8b": llm_summarization_mt.MDS_WikinewsSum_ClusterChunk_5_Cluster_Size,
    "MDS_WikinewsClust10R_42_8b": llm_summarization_mt.MDS_WikinewsSum_ClusterChunk_10_Random,
    "MDS_WikinewsClust10O_42_8b": llm_summarization_mt.MDS_WikinewsSum_ClusterChunk_10_Original_Order,
    "MDS_WikinewsClust10C_42_8b": llm_summarization_mt.MDS_WikinewsSum_ClusterChunk_10_Cluster_Size,
    "MDS_WikinewsSent1L00_42_8b": llm_summarization_mt.MDS_WikinewsSum_SentenceChunk_1_00_512_sbert_comparison,
    "MDS_WikinewsSent1L05_42_8b": llm_summarization_mt.MDS_WikinewsSum_SentenceChunk_1_05_512_sbert_comparison,
    "MDS_WikinewsSent3L00_42_8b": llm_summarization_mt.MDS_WikinewsSum_SentenceChunk_3_00_512_sbert_comparison,
    "MDS_WikinewsSent3L05_42_8b": llm_summarization_mt.MDS_WikinewsSum_SentenceChunk_3_05_512_sbert_comparison,
    "MDS_WikinewsSent5L00_42_8b": llm_summarization_mt.MDS_WikinewsSum_SentenceChunk_5_00_512_sbert_comparison,
    "MDS_WikinewsSent5L05_42_8b": llm_summarization_mt.MDS_WikinewsSum_SentenceChunk_5_05_512_sbert_comparison,
    "MDS_WikinewsSent10L00_42_8b": llm_summarization_mt.MDS_WikinewsSum_SentenceChunk_10_00_512_sbert_comparison,
    "MDS_WikinewsSent10L05_42_8b": llm_summarization_mt.MDS_WikinewsSum_SentenceChunk_10_05_512_sbert_comparison,
    "SummFewshot0_20minTS250_1_8b": llm_summarization_mt.SummFewshot_250TestSample_20Minuten,
    "SummFewshot1_20minTS250_1_8b": llm_summarization_mt.SummFewshot_250TestSample_20Minuten,
    "SummFewshot2_20minTS250_1_8b": llm_summarization_mt.SummFewshot_250TestSample_20Minuten,
    "SummFewshot0_20minTS250_2_8b": llm_summarization_mt.SummFewshot_250TestSample_20Minuten,
    "SummFewshot1_20minTS250_2_8b": llm_summarization_mt.SummFewshot_250TestSample_20Minuten,
    "SummFewshot2_20minTS250_2_8b": llm_summarization_mt.SummFewshot_250TestSample_20Minuten,
    "SummFewshot0_20min0_5_8b": llm_summarization_mt.SummShard0_20Minuten,
    "SummFewshot1_20min0_5_8b": llm_summarization_mt.SummShard0_20Minuten,
    "SummFewshot2_20min0_5_8b": llm_summarization_mt.SummShard0_20Minuten,
    "SummFewshot0_20min0_2_8b": llm_summarization_mt.SummShard0_20Minuten,
    "SummFewshot1_20min0_2_8b": llm_summarization_mt.SummShard0_20Minuten,
    "SummFewshot2_20min0_2_8b": llm_summarization_mt.SummShard0_20Minuten,
    "SummFewshot0_20Minuten_1": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot0_20Minuten_2": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot0_20Minuten_3": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot0_20Minuten_4": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot0_20Minuten_5": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot1_20Minuten_1": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot1_20Minuten_2": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot1_20Minuten_3": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot1_20Minuten_4": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot1_20Minuten_5": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot2_20Minuten_1": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot2_20Minuten_2": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot2_20Minuten_3": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot2_20Minuten_4": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot2_20Minuten_5": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot4_20Minuten_1": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot4_20Minuten_2": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot4_20Minuten_3": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot4_20Minuten_4": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot4_20Minuten_5": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot8_20Minuten_1": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot8_20Minuten_2": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot8_20Minuten_3": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot8_20Minuten_4": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot8_20Minuten_5": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot0_20Minuten_1_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot0_20Minuten_2_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot0_20Minuten_3_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot0_20Minuten_4_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot0_20Minuten_5_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot1_20Minuten_1_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot1_20Minuten_2_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot1_20Minuten_3_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot1_20Minuten_4_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot1_20Minuten_5_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot2_20Minuten_1_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot2_20Minuten_2_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot2_20Minuten_3_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot2_20Minuten_4_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot2_20Minuten_5_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot4_20Minuten_1_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot4_20Minuten_2_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot4_20Minuten_3_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot4_20Minuten_4_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot4_20Minuten_5_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot8_20Minuten_1_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot8_20Minuten_2_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot8_20Minuten_3_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot8_20Minuten_4_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummFewshot8_20Minuten_5_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummLtM_20Minuten": llm_summarization_mt.SummLtM_20Minuten,
    "SummLtMDe_20Minuten": llm_summarization_mt.SummLtMDe_20Minuten,
    "SummSmolSample_20Minuten_1": llm_summarization_mt.SummSampleSmolSmol_20Minuten,
    "SummPalm2_20Minuten_2": llm_summarization_mt.SummSample_BadPalm2Examples_20Minuten,
    "SummSample_20Minuten_1": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_2": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_3": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_4": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_5": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_1_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_2_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_3_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_4_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_5_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_40_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_41_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_42_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_23_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_7_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_9_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_11_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_13_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_15_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_17_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_19_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_22_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_43_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_44_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_45_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_46_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_47_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_48_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_49_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_40": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_41": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_42": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_23": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_7": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_9": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_11": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_13": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_15": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_17": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_19": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_22": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_43": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_44": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_45": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_46": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_47": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_48": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummSample_20Minuten_49": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummNonEmpty_20minSmol_1": llm_summarization_mt.SummShard0_20Minuten_NonEmpty,
    "SummEmpty_20minSmol_1_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummEmpty_20minSmol_2_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummEmpty_20minSmol_4_8b": llm_summarization_mt.SummSampleSmol_20Minuten,
    "SummNonEmpty_20minSmol_1_8b": llm_summarization_mt.SummShard0_20Minuten_NonEmpty,
    "SummNonEmpty_20minSmol_2_8b": llm_summarization_mt.SummShard0_20Minuten_NonEmpty,
    "SummNonEmpty_20minSmol_4_8b": llm_summarization_mt.SummShard0_20Minuten_NonEmpty,
    "SummarizationTask_20min0_1_8b": llm_summarization_mt.SummShard0_20Minuten,
    "SummarizationTask_20min0_2_8b": llm_summarization_mt.SummShard0_20Minuten,
    "SummarizationTask_20min0_3_8b": llm_summarization_mt.SummShard0_20Minuten,
    "SummarizationTask_20min0_4_8b": llm_summarization_mt.SummShard0_20Minuten,
    "SummarizationTask_20min0_5_8b": llm_summarization_mt.SummShard0_20Minuten,
    "SummarizationTask_20min1_1_8b": llm_summarization_mt.SummShard1_20Minuten,
    "SummarizationTask_20min1_2_8b": llm_summarization_mt.SummShard1_20Minuten,
    "SummarizationTask_20min1_3_8b": llm_summarization_mt.SummShard1_20Minuten,
    "SummarizationTask_20min1_4_8b": llm_summarization_mt.SummShard1_20Minuten,
    "SummarizationTask_20min1_5_8b": llm_summarization_mt.SummShard1_20Minuten,
    "SummarizationTask_20min2_1_8b": llm_summarization_mt.SummShard2_20Minuten,
    "SummarizationTask_20min2_2_8b": llm_summarization_mt.SummShard2_20Minuten,
    "SummarizationTask_20min2_3_8b": llm_summarization_mt.SummShard2_20Minuten,
    "SummarizationTask_20min2_4_8b": llm_summarization_mt.SummShard2_20Minuten,
    "SummarizationTask_20min2_5_8b": llm_summarization_mt.SummShard2_20Minuten,
    "SummarizationTask_20min3_1_8b": llm_summarization_mt.SummShard3_20Minuten,
    "SummarizationTask_20min3_2_8b": llm_summarization_mt.SummShard3_20Minuten,
    "SummarizationTask_20min3_3_8b": llm_summarization_mt.SummShard3_20Minuten,
    "SummarizationTask_20min3_4_8b": llm_summarization_mt.SummShard3_20Minuten,
    "SummarizationTask_20min3_5_8b": llm_summarization_mt.SummShard3_20Minuten,
    "SummarizationTask_20Minuten": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_1": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_2": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_3": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_4": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_5": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_6": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_7": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_8": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_9": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_10": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_11": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_12": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_13": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_14": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_15": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_16": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_17": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_18": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_19": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_20": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_1_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_2_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_3_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_4_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_5_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_6_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_7_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_8_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_9_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_10_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_11_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_12_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_13_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_14_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_15_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_16_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_17_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_18_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_19_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_20_8b": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_1": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_2": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_3": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_4": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_5": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_6": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_7": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_8": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_9": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_10": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_11": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_12": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_13": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_14": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_15": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_16": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_17": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_18": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_19": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T01_20": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_1": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_2": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_3": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_4": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_5": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_6": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_7": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_8": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_9": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_10": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_11": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_12": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_13": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_14": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_15": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_16": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_17": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_18": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_19": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T05_20": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_1": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_2": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_3": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_4": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_5": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_6": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_7": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_8": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_9": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_10": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_11": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_12": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_13": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_14": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_15": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_16": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_17": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_18": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_19": llm_summarization_mt.SummarizationTask_20Minuten,
    "SummarizationTask_20Minuten_T10_20": llm_summarization_mt.SummarizationTask_20Minuten,
    "seahorse_classification": seahorse_classification.SeahorseClassificationTask,
    "seahorse_manual_test": seahorse_classification_automatic.AutomaticSeahorseClassificationTask_Local,
    "seahorse_base_datasets_100": seahorse_classification_automatic.AutomaticSeahorseClassificationTask_BaseDatasets,
    "seahorse_fco_experiments_100": seahorse_classification_automatic.AutomaticSeahorseClassificationTask_FCOExperiments,
    # Requires manual download of data.
    # "storycloze_2016": storycloze.StoryCloze2016,
    # "storycloze_2018": storycloze.StoryCloze2018,
    # "sat": sat.SATAnalogies,
    **xcopa.construct_tasks(),
    **bigbench.create_all_tasks(),
    **xstorycloze.create_all_tasks(),
    **xwinograd.create_all_tasks(),
    **pawsx.construct_tasks(),
    **xnli.construct_tasks(),
    **mgsm.construct_tasks(),
}

ALL_TASKS = sorted(list(TASK_REGISTRY))

_EXAMPLE_JSON_PATH = "split:key:/absolute/path/to/data.json"


def add_json_task(task_name):
    """Add a JSON perplexity task if the given task name matches the
    JSON task specification.

    See `json.JsonPerplexity`.
    """
    if not task_name.startswith("json"):
        return

    def create_json_task():
        splits = task_name.split("=", 1)
        if len(splits) != 2 or not splits[1]:
            raise ValueError(
                "json tasks need a path argument pointing to the local "
                "dataset, specified like this: json="
                + _EXAMPLE_JSON_PATH
                + ' (if there are no splits, use "train")'
            )

        json_path = splits[1]
        if json_path == _EXAMPLE_JSON_PATH:
            raise ValueError(
                "please do not copy the example path directly, but substitute "
                "it with a path to your local dataset"
            )
        return lambda: json.JsonPerplexity(json_path)

    TASK_REGISTRY[task_name] = create_json_task()


def get_task(task_name):
    try:
        add_json_task(task_name)
        return TASK_REGISTRY[task_name]
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def load_prompt(task_name, model_id, prompt_version):
    """Load the appropriate prompt based on task name, model prefix, and version."""
    file_path = os.path.join("configs", "prompt_templates", f"{task_name}.json")

    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r') as f:
        all_prompts = json.load(f)

    # Find the model name starting with the given prefix
    matched_model_name = next((name for name in all_prompts.keys() if model_id.startswith(name)), None)

    if not matched_model_name:
        print(
            f"Did not find a prompt for the model with id {model_id} and prompt version {prompt_version}. Using "
            f"default prompt for task {task_name}")
        return None
    matched_prompt = all_prompts[matched_model_name].get(prompt_version, None)
    if not matched_prompt:
        print(
            f"Did not find a prompt for the given prompt version {prompt_version}. Using default prompt for task {task_name}")
        return None
    return matched_prompt


def get_task_dict(task_name_list: List[Union[str, lm_eval.base.Task]], model_id: str,
                  prompt_version_per_task: str = None):
    prompt_templates = {}

    if prompt_version_per_task:
        prompt_versions = prompt_version_per_task.split(",")
        assert len(prompt_versions) == len(task_name_list), "Mismatch in number of tasks and provided prompt versions."

        for task_name, prompt_version in zip(task_name_list, prompt_versions):
            prompt = load_prompt(task_name, model_id, prompt_version)
            if prompt:
                prompt_templates[task_name] = prompt
                print(f"Selected prompt version: {prompt_version} for task {task_name}.\nFinal prompt: {prompt}")

    task_name_dict = {
        task_name: get_task(task_name)(prompt_template=prompt_templates.get(task_name, None))
        for task_name in task_name_list
        if isinstance(task_name, str)
    }

    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list
        if not isinstance(task_object, str)
    }

    assert set(task_name_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys())), \
        "Task name collision between string and object tasks."

    return {**task_name_dict, **task_name_from_object_dict}
