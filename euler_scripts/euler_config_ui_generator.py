import json
import os
import re
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk, messagebox
from typing import List, Dict, Tuple
from subprocess import call
import euler_scripts.euler_config_generators_and_run_manager as config_generator
from lm_eval.utils import TaskConfig


@dataclass
class UIConfigValues:
    base_config_dir: str
    model_name_values: List[str]
    few_shot_values: List[int]
    tasks_with_prompt_templates: Dict[str, str]
    seeds: List[int]
    few_shot_sampling_strategies: List[str]


def read_json(json_file_name: str):
    with open(json_file_name, 'r') as file:
        json_file = json.load(file)
    return json_file


def get_model_prompt_versions(model_name: str, prompt_template_json: Dict) -> List:
    for prompt_model_name, prompts in prompt_template_json.items():
        if prompt_model_name in model_name:
            return list(prompts.keys())
    print(f"The model {model_name} has no corresponding prompts defined in the template")
    return []


# def get_task_name_from_prompt_template(selected_prompt_template: str):
#     return os.path.splitext(os.path.basename(selected_prompt_template))[0]


class EulerConfigAndRunUIManager:
    def __init__(self, config_values_file: str):
        self.root = tk.Tk()
        self.root.title("Config Generator")
        # Bind the close button event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.config_ui_values = self.init_config_values(config_values_file)

        self.current_ui_row = 0

        # Model names label and multi-select Listbox
        default_width = 50
        long_width = 100
        default_height = 5
        long_height = 10
        self.model_listbox = self.generate_list_box(label_text="Model Names", select_mode=tk.MULTIPLE,
                                                    box_width=long_width,
                                                    box_height=long_height,
                                                    box_values=self.config_ui_values.model_name_values)
        self.model_listbox.bind('<<ListboxSelect>>', self.update_prompt_version_listbox)

        # Fewshot label and multi-select Listbox
        self.fewshot_listbox = self.generate_list_box(label_text="Fewshots:", select_mode=tk.MULTIPLE,
                                                      box_width=default_width,
                                                      box_height=default_height,
                                                      box_values=self.config_ui_values.few_shot_values)

        # Seed label and multi-select Listbox
        self.fewshot_strategies_listbox = self.generate_list_box(label_text="Fewshot strategies:",
                                                                 select_mode=tk.MULTIPLE,
                                                                 box_width=default_width, box_height=default_height,
                                                                 box_values=self.config_ui_values.few_shot_sampling_strategies)

        # Seed label and multi-select Listbox
        self.seed_listbox = self.generate_list_box(label_text="Seeds:", select_mode=tk.MULTIPLE,
                                                   box_width=default_width,
                                                   box_height=default_height, box_values=self.config_ui_values.seeds)

        self.tasks_listbox = self.generate_list_box(label_text="Tasks:", select_mode=tk.SINGLE,
                                                    box_width=long_width,
                                                    box_height=default_height,
                                                    box_values=list(
                                                        self.config_ui_values.tasks_with_prompt_templates.keys()))
        self.tasks_listbox.bind('<<ListboxSelect>>', self.update_prompt_version_listbox)

        self.selected_prompt_versions_listbox = self.generate_list_box(label_text="Prompts:",
                                                                       select_mode=tk.MULTIPLE,
                                                                       box_width=long_width,
                                                                       box_height=default_height,
                                                                       box_values=[])

        self.load_in_8bit_checkbox_value = self.generate_checkbox("Load in 8bit")
        self.use_flash_attention_2_checkbox_value = self.generate_checkbox("Use Flash Attention 2")

        # Execute button
        self.create_config_files_button = ttk.Button(self.root, text="Create Config Files",
                                                     command=self.execute_create_config_files_script)
        self.create_config_files_button.grid(row=self.current_ui_row, column=0, columnspan=3, pady=20)
        self.current_ui_row += 1

        self.run_euler_jobs_button = ttk.Button(self.root, text="Run euler jobs",
                                                command=self.execute_run_euler_jobs_script)
        self.run_euler_jobs_button.grid(row=self.current_ui_row, column=0, columnspan=3, pady=20)
        self.current_ui_row += 1

        # Output label to show the script execution result
        self.output_label = ttk.Label(self.root, text="")
        self.output_label.grid(row=self.current_ui_row, column=0, columnspan=3, pady=5)
        self.current_ui_row += 1
        self.root.mainloop()

    @staticmethod
    def init_config_values(config_file: str) -> UIConfigValues:
        config_values = read_json(json_file_name=config_file)
        ui_config_values = UIConfigValues(
            base_config_dir=config_values["base_config_dir"],
            model_name_values=config_values["model_checkpoints"],
            few_shot_values=list(range(config_values["few_shot_values"]["lower_range"],
                                       config_values["few_shot_values"]["upper_range"])),
            tasks_with_prompt_templates=config_values["tasks_with_prompt_templates"],
            seeds=config_values["seeds"],
            few_shot_sampling_strategies=config_values["few_shot_sampling_strategies"]
        )

        return ui_config_values

    def generate_list_box(self, label_text: str, select_mode: str, box_width: int, box_height: int, box_values: List):
        box_label = ttk.Label(self.root, text=label_text)
        box_label.grid(row=self.current_ui_row, column=0, sticky='w', padx=5, pady=5)
        listbox = tk.Listbox(self.root, selectmode=select_mode, exportselection=0, width=box_width, height=box_height)
        listbox.grid(row=self.current_ui_row, column=1, padx=5, pady=5)
        self.current_ui_row += 1
        # Pre-fill the listbox with default values
        for value in box_values:
            listbox.insert(tk.END, value)
        return listbox

    def generate_checkbox(self, checkbox_text: str, default: bool = False):
        # Create a Checkbutton
        checkbox_value = tk.BooleanVar()
        checkbox_value.set(default)
        checkbox = tk.Checkbutton(self.root, text=checkbox_text, variable=checkbox_value)
        checkbox.grid(row=self.current_ui_row, column=1, padx=5, pady=5)
        self.current_ui_row += 1
        return checkbox_value

    def update_prompt_version_listbox(self, event):
        # Clear the current items from the dynamic_listbox
        self.selected_prompt_versions_listbox.delete(0, tk.END)
        tasks_names = [self.tasks_listbox.get(i) for i in self.tasks_listbox.curselection()]
        if len(tasks_names) == 0:
            return
        selected_task = tasks_names[0]
        selected_prompt_template = self.config_ui_values.tasks_with_prompt_templates[selected_task]
        possible_prompt_versions = None
        if selected_prompt_template == "default":
            possible_prompt_versions = [selected_prompt_template]
        else:
            model_names = [self.model_listbox.get(i) for i in self.model_listbox.curselection()]
            if len(model_names) == 0:
                return
            prompt_template_json = read_json(json_file_name=selected_prompt_template)
            # Iterate over selected items in model_listbox
            for model_name in model_names:

                prompt_versions = get_model_prompt_versions(model_name=model_name,
                                                            prompt_template_json=prompt_template_json)
                if not possible_prompt_versions:
                    possible_prompt_versions = set(prompt_versions)
                else:
                    possible_prompt_versions.intersection_update(prompt_versions)

        for item in possible_prompt_versions:
            self.selected_prompt_versions_listbox.insert(tk.END, item)

    def execute_create_config_files_script(self):
        # Fetch the values from UI
        model_names = [self.model_listbox.get(i) for i in self.model_listbox.curselection()]
        fewshots = [int(self.fewshot_listbox.get(i)) for i in self.fewshot_listbox.curselection()]
        seeds = [int(self.seed_listbox.get(i)) for i in self.seed_listbox.curselection()]
        few_shot_sampling_strategies = [self.fewshot_strategies_listbox.get(i) for i in
                                        self.fewshot_strategies_listbox.curselection()]
        selected_task = [self.tasks_listbox.get(i) for i in
                         self.tasks_listbox.curselection()][0]
        selected_prompt_version_names = [self.selected_prompt_versions_listbox.get(i) for i in
                                         self.selected_prompt_versions_listbox.curselection()]
        selected_prompt_template = self.config_ui_values.tasks_with_prompt_templates[selected_task]
        task_config_list = [
            TaskConfig(task_name=selected_task, prompt_template=selected_prompt_template, prompt_version=prompt_version)
            for prompt_version in selected_prompt_version_names]

        if len(model_names) == 0 or len(fewshots) == 0 or len(seeds) == 0 or len(
                few_shot_sampling_strategies) == 0 or len(
            selected_prompt_version_names) == 0:
            messagebox.showwarning("Warning", "Please select at least one value from each of the boxes!")
            return

        selected_prompt_version_names = [(selected_task, prompt_version) for prompt_version in
                                         selected_prompt_version_names]
        # You can set these values in the config_generator.py script and then run its main()
        config_generator.base_yaml_config_path = self.config_ui_values.base_config_dir
        config_generator.model_names = model_names
        config_generator.num_fewshots = fewshots
        # config_generator.selected_tasks_with_prompt_version_names = selected_prompt_version_names
        config_generator.task_config_list = task_config_list
        config_generator.seeds = seeds
        config_generator.few_shot_sampling_strategies = few_shot_sampling_strategies
        config_generator.load_in_8bit = self.load_in_8bit_checkbox_value.get()
        config_generator.use_flash_attention_2 = self.use_flash_attention_2_checkbox_value.get()

        config_generator.main()

        # Provide feedback
        self.output_label.config(text="Script executed successfully!")

    def execute_run_euler_jobs_script(self):
        # Fetch selected values
        model_names = [self.model_listbox.get(i) for i in self.model_listbox.curselection()]
        fewshots = [str(self.fewshot_listbox.get(i)) for i in self.fewshot_listbox.curselection()]
        seeds = [str(self.seed_listbox.get(i)) for i in self.seed_listbox.curselection()]
        few_shot_sampling_strategies = [self.fewshot_strategies_listbox.get(i) for i in
                                        self.fewshot_strategies_listbox.curselection()]
        selected_task = [self.tasks_listbox.get(i) for i in
                         self.tasks_listbox.curselection()][0]

        if len(model_names) == 0 or len(fewshots) == 0 or len(seeds) == 0 or len(few_shot_sampling_strategies) == 0:
            messagebox.showwarning("Warning", "Please select at least one value from each of the boxes!")
            return
        # Convert lists to space-separated strings
        models_str = " ".join(model_names)
        num_shots_str = " ".join(fewshots)
        strategies_str = " ".join(few_shot_sampling_strategies)
        seeds_str = " ".join(seeds)
        base_config_dir = self.config_ui_values.base_config_dir

        # Call the bash script with the values as arguments
        call(['euler_scripts/euler_run_scheduler.sh', models_str, num_shots_str, strategies_str, seeds_str,
              selected_task, base_config_dir])

    def on_closing(self):
        # Any other cleanup tasks you might want to do can be added here
        self.root.destroy()  # This will close the application


def main():
    config_values_file = "euler_scripts/configs/faithfulness_model_configs.json"
    #config_values_file = "euler_scripts/configs/domain_adaptation_summarization_model_configs.json"  # "euler_scripts/configs/faithfulness_model_configs.json"
    EulerConfigAndRunUIManager(config_values_file=config_values_file)


if __name__ == "__main__":
    main()
