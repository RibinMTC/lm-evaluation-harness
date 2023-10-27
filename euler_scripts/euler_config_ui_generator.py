import json
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict
from subprocess import call
import euler_scripts.euler_config_generators_and_run_manager as config_generator

MODEL_NAME_VALUES = ["LeoLM/leo-hessianai-13b", "google/flan-ul2", "LeoLM/leo-hessianai-7b",
                     "LeoLM/leo-mistral-hessianai-7b", "NousResearch/Llama-2-7b-hf", "NousResearch/Llama-2-13b-hf",
                     "mtc/LeoLM-leo-hessianai-13b-all-labels-classification-english-one-epoch-qlora-4bit-merged",
                     "mtc/LeoLM-leo-hessianai-13b-all-labels-classification-one-epoch-qlora-4bit-merged"]
FEW_SHOT_VALUES = list(range(32))
PROMPT_TEMPLATES_VALUES = ["configs/prompt_templates/faithfulness_benchmark_final_swisstext23_benchmark.json",
                           "configs/prompt_templates/faithfulness_benchmark_final_swisstext23_multi_label.json"]
SEEDS = [5, 10, 42, 100]
FEW_SHOT_SAMPLING_STRATEGIES = ["stratified", "packed"]
current_ui_row = 0


def read_json(json_file_name: str):
    with open(json_file_name, 'r') as file:
        json_file = json.load(file)
    return json_file


def get_model_prompt_versions(model_name: str, prompt_template_json: Dict) -> List:
    for prompt_model_name, prompts in prompt_template_json.items():
        if prompt_model_name in model_name:
            return list(prompts.keys())
    raise ValueError(f"The model {model_name} has no corresponding prompts defined in the template")


def generate_list_box(label_text: str, select_mode: str, box_width: int, box_height: int, box_values: List):
    global current_ui_row
    box_label = ttk.Label(root, text=label_text)
    box_label.grid(row=current_ui_row, column=0, sticky='w', padx=5, pady=5)
    listbox = tk.Listbox(root, selectmode=select_mode, exportselection=0, width=box_width, height=box_height)
    listbox.grid(row=current_ui_row, column=1, padx=5, pady=5)
    current_ui_row += 1
    # Pre-fill the listbox with default values
    for value in box_values:
        listbox.insert(tk.END, value)
    return listbox


def update_prompt_version_listbox(event):
    # Clear the current items from the dynamic_listbox
    selected_prompt_versions_listbox.delete(0, tk.END)
    prompt_template_names = [prompt_template_listbox.get(i) for i in prompt_template_listbox.curselection()]
    model_names = [model_listbox.get(i) for i in model_listbox.curselection()]
    if len(prompt_template_names) == 0 or len(model_names) == 0:
        return
    prompt_template_json = read_json(json_file_name=prompt_template_names[0])
    # Iterate over selected items in model_listbox
    common_model_prompt_versions = None
    for model_name in model_names:

        prompt_versions = get_model_prompt_versions(model_name=model_name, prompt_template_json=prompt_template_json)
        if not common_model_prompt_versions:
            common_model_prompt_versions = set(prompt_versions)
        else:
            common_model_prompt_versions.intersection_update(prompt_versions)

    for item in common_model_prompt_versions:
        selected_prompt_versions_listbox.insert(tk.END, item)


def execute_create_config_files_script():
    # Fetch the values from UI
    model_names = [model_listbox.get(i) for i in model_listbox.curselection()]
    fewshots = [int(fewshot_listbox.get(i)) for i in fewshot_listbox.curselection()]
    seeds = [int(seed_listbox.get(i)) for i in seed_listbox.curselection()]
    few_shot_sampling_strategies = [fewshot_strategies_listbox.get(i) for i in
                                    fewshot_strategies_listbox.curselection()]
    selected_prompt_version_names = [selected_prompt_versions_listbox.get(i) for i in
                                     selected_prompt_versions_listbox.curselection()]

    if len(model_names) == 0 or len(fewshots) == 0 or len(seeds) == 0 or len(few_shot_sampling_strategies) == 0 or len(
            selected_prompt_version_names) == 0:
        messagebox.showwarning("Warning", "Please select at least one value from each of the boxes!")
        return

    # You can set these values in the config_generator.py script and then run its main()
    config_generator.model_names = model_names
    config_generator.num_fewshots = fewshots
    config_generator.selected_prompt_version_names = selected_prompt_version_names
    config_generator.seeds = seeds
    config_generator.few_shot_sampling_strategies = few_shot_sampling_strategies

    config_generator.main()

    # Provide feedback
    output_label.config(text="Script executed successfully!")


def execute_run_euler_jobs_script():
    # Fetch selected values
    model_names = [model_listbox.get(i) for i in model_listbox.curselection()]
    fewshots = [str(fewshot_listbox.get(i)) for i in fewshot_listbox.curselection()]
    seeds = [str(seed_listbox.get(i)) for i in seed_listbox.curselection()]
    few_shot_sampling_strategies = [fewshot_strategies_listbox.get(i) for i in
                                    fewshot_strategies_listbox.curselection()]

    if len(model_names) == 0 or len(fewshots) == 0 or len(seeds) == 0 or len(few_shot_sampling_strategies) == 0:
        messagebox.showwarning("Warning", "Please select at least one value from each of the boxes!")
        return
    # Convert lists to space-separated strings
    models_str = " ".join(model_names)
    num_shots_str = " ".join(fewshots)
    strategies_str = " ".join(few_shot_sampling_strategies)
    seeds_str = " ".join(seeds)

    # Call the bash script with the values as arguments
    call(['euler_scripts/euler_run_scheduler.sh', models_str, num_shots_str, strategies_str, seeds_str])


def on_closing():
    # Any other cleanup tasks you might want to do can be added here
    root.destroy()  # This will close the application


root = tk.Tk()
root.title("Config Generator")
# Bind the close button event
root.protocol("WM_DELETE_WINDOW", on_closing)

# Model names label and multi-select Listbox
default_width = 50
long_width = 100
default_height = 5
long_height = 10
model_listbox = generate_list_box(label_text="Model Names", select_mode=tk.MULTIPLE, box_width=default_width,
                                  box_height=long_height,
                                  box_values=MODEL_NAME_VALUES)
model_listbox.bind('<<ListboxSelect>>', update_prompt_version_listbox)

# Fewshot label and multi-select Listbox
fewshot_listbox = generate_list_box(label_text="Fewshots:", select_mode=tk.MULTIPLE, box_width=default_width,
                                    box_height=default_height,
                                    box_values=FEW_SHOT_VALUES)

# Seed label and multi-select Listbox
fewshot_strategies_listbox = generate_list_box(label_text="Fewshot strategies:", select_mode=tk.MULTIPLE,
                                               box_width=default_width, box_height=default_height,
                                               box_values=FEW_SHOT_SAMPLING_STRATEGIES)

# Seed label and multi-select Listbox
seed_listbox = generate_list_box(label_text="Seeds:", select_mode=tk.MULTIPLE, box_width=default_width,
                                 box_height=default_height, box_values=SEEDS)

prompt_template_listbox = generate_list_box(label_text="Prompt Template:", select_mode=tk.SINGLE, box_width=long_width,
                                            box_height=default_height, box_values=PROMPT_TEMPLATES_VALUES)
prompt_template_listbox.bind('<<ListboxSelect>>', update_prompt_version_listbox)

selected_prompt_versions_listbox = generate_list_box(label_text="Prompts:", select_mode=tk.MULTIPLE,
                                                     box_width=default_width, box_height=default_height, box_values=[])

# Execute button
create_config_files_button = ttk.Button(root, text="Create Config Files", command=execute_create_config_files_script)
create_config_files_button.grid(row=current_ui_row, column=0, columnspan=3, pady=20)
current_ui_row += 1

run_euler_jobs_button = ttk.Button(root, text="Run euler jobs", command=execute_run_euler_jobs_script)
run_euler_jobs_button.grid(row=current_ui_row, column=0, columnspan=3, pady=20)
current_ui_row += 1

# Output label to show the script execution result
output_label = ttk.Label(root, text="")
output_label.grid(row=current_ui_row, column=0, columnspan=3, pady=5)
current_ui_row += 1
root.mainloop()
