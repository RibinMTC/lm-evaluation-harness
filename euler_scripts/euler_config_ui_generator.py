import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Any

from euler_config_generation_utils import *

from lm_eval.utils import TaskConfig


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

        # Max Length values
        self.max_lengths_listbox = self.generate_list_box(label_text="Max Context Length:", select_mode=tk.SINGLE,
                                                          box_width=default_width,
                                                          box_height=default_height,
                                                          box_values=self.config_ui_values.max_context_length_values,
                                                          )

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

        self.batch_size_listbox = self.generate_list_box(label_text="Batch Size:",
                                                         select_mode=tk.SINGLE,
                                                         box_width=default_width,
                                                         box_height=default_height,
                                                         box_values=["1", "2", "4", "8", "16", "32", "64"])

        validate_command = (self.root.register(self.is_number), '%P')
        self.start_range_entry = self.generate_entry(label_description="Start Range", validate_command=validate_command)

        self.end_range_entry = self.generate_entry(label_description="End Range", validate_command=validate_command)

        self.load_in_8bit_checkbox_value = self.generate_checkbox("Load in 8bit")
        self.use_flash_attention_2_checkbox_value = self.generate_checkbox("Use Flash Attention 2")

        # Execute button
        self.create_config_files_button = ttk.Button(self.root, text="Create Config Files",
                                                     command=self.execute_create_config_files_script)
        self.create_config_files_button.grid(row=self.current_ui_row, column=0, columnspan=3, pady=20)
        self.current_ui_row += 1

        # Output label to show the script execution result
        self.output_label = ttk.Label(self.root, text="")
        self.output_label.grid(row=self.current_ui_row, column=0, columnspan=3, pady=5)
        self.current_ui_row += 1
        self.root.mainloop()

    @staticmethod
    def is_number(input):
        # Allow only numeric input or empty input (for None)
        if input.isdigit() or input == "":
            return True
        return False

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
            few_shot_sampling_strategies=config_values["few_shot_sampling_strategies"],
            max_context_length_values=config_values["max_context_length_values"]
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

    def generate_entry(self, label_description: str, validate_command: Any):
        entry_label = ttk.Label(self.root, text=label_description)
        entry = tk.Entry(self.root, validate="key", validatecommand=validate_command)
        entry_label.grid(row=self.current_ui_row, column=0, sticky='w', padx=5, pady=5)
        entry.grid(row=self.current_ui_row, column=1, padx=5, pady=5)
        self.current_ui_row += 1
        return entry

    def update_prompt_version_listbox(self, event):
        # Clear the current items from the dynamic_listbox
        self.selected_prompt_versions_listbox.delete(0, tk.END)
        tasks_names = [self.tasks_listbox.get(i) for i in self.tasks_listbox.curselection()]
        if len(tasks_names) == 0:
            return
        selected_task = tasks_names[0]
        selected_prompt_template = self.config_ui_values.tasks_with_prompt_templates[selected_task]
        possible_prompt_versions = None
        if not selected_prompt_template:
            possible_prompt_versions = ["default"]
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

        max_context_length_selection = self.max_lengths_listbox.curselection()
        max_context_length = None
        if max_context_length_selection:
            max_context_length = int(self.max_lengths_listbox.get(max_context_length_selection[0]))

        if len(model_names) == 0 or len(fewshots) == 0 or len(seeds) == 0 or len(
                few_shot_sampling_strategies) == 0 or len(selected_prompt_version_names) == 0:
            messagebox.showwarning("Warning", "Please select at least one value from each of the boxes!")
            return

        selected_batch_index = self.batch_size_listbox.curselection()[0]
        selected_batch_size = self.batch_size_listbox.get(selected_batch_index)
        start_range = int(self.start_range_entry.get()) if self.start_range_entry.get() else None
        end_range = int(self.end_range_entry.get()) if self.end_range_entry.get() else None

        # You can set these values in the config_generator.py script and then run its main()
        selected_ui_config_values = SelectedUIConfigValues(
            base_config_dir=self.config_ui_values.base_config_dir,
            model_name_values=model_names,
            few_shot_values=fewshots,
            tasks_config_list=task_config_list,
            seeds=seeds,
            few_shot_sampling_strategies=few_shot_sampling_strategies,
            load_in_8bit=self.load_in_8bit_checkbox_value.get(),
            use_flash_attention_2=self.use_flash_attention_2_checkbox_value.get(),
            max_context_length=max_context_length,
            selected_batch_size=selected_batch_size,
            start_range=start_range,
            end_range=end_range)

        save_config_files(selected_ui_config_values=selected_ui_config_values)

        # Provide feedback
        self.output_label.config(text="Script executed successfully!")

    def on_closing(self):
        # Any other cleanup tasks you might want to do can be added here
        self.root.destroy()  # This will close the application


def main():
    config_values_file = "euler_scripts/configs/faithfulness_model_configs.json"
    # config_values_file = "euler_scripts/configs/domain_adaptation_summarization_model_configs.json"  # "euler_scripts/configs/faithfulness_model_configs.json"
    EulerConfigAndRunUIManager(config_values_file=config_values_file)


if __name__ == "__main__":
    main()
