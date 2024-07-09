import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Any
from subprocess import call
from euler_config_generation_utils import *

from lm_eval.utils import TaskConfig


class EulerConfigValues:
    euler_gpu: str
    euler_runtime: str
    num_gpus: int
    num_cpus: int
    memory_per_cpu_gb: int


# Define the mapping of possible values for each field
EULER_CONFIG_POSSIBLE_VALUES = {
    'euler_gpu': ["rtx_3090", "a100_80gb", "rtx_4090", "v100", "a100-pcie-40gb"],
    'euler_runtime': ["00:30", "01:00", "02:00", "04:00", "08:00", "12:00", "24:00", "48:00"],
    'num_gpus': [1, 2, 4, 8],
    'num_cpus': [2, 4, 8, 16],
    'memory_per_cpu_gb': [4, 8, 12, 16, 32, 64]
}

EULER_CONFIG_DEFAULT_VALUES = {
    'euler_gpu': "rtx_3090",
    'euler_runtime': "01:00",
    'num_gpus': 1,
    'num_cpus': 4,
    'memory_per_cpu_gb': 12
}

# Map EulerConfigValues fields to JSON fields
EULER_CONFIG_FIELD_MAPPING = {
    'euler_gpu': 'gpu',
    'euler_runtime': 'run_duration_hours',
    'num_gpus': 'num_gpus',
    'num_cpus': 'num_cpus',
    'memory_per_cpu_gb': 'memory_per_cpu_gb'
}

EULER_JSON_CONFIG_PATH = "euler_scripts/lm_eval_euler_config.json"


class EulerConfigRunUIManager:
    def __init__(self, root_config_dir: str):
        self.task_hierarchy, self.config_name_to_path_map = self.list_files_by_task(root_config_dir)
        self.root = tk.Tk()
        self.root.title("Config Generator")
        # Bind the close button event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.current_ui_row = 0

        # Model names label and multi-select Listbox
        default_width = 50
        long_width = 100
        default_height = 5
        long_height = 10

        self.tasks_listbox = self.generate_list_box(label_text="Tasks:", select_mode=tk.MULTIPLE,
                                                    box_width=long_width,
                                                    box_height=long_height,
                                                    box_values=list(
                                                        self.task_hierarchy.keys()))

        self.tasks_listbox.bind('<<ListboxSelect>>', self.update_models_listbox)

        self.models_listbox = self.generate_list_box(label_text="Models:",
                                                     select_mode=tk.MULTIPLE,
                                                     box_width=long_width,
                                                     box_height=long_height,
                                                     box_values=[])

        self.models_listbox.bind('<<ListboxSelect>>', self.update_configs_listbox)

        self.configs_listbox = self.generate_list_box(label_text="Configs:",
                                                      select_mode=tk.MULTIPLE,
                                                      box_width=long_width,
                                                      box_height=long_height,
                                                      box_values=[])

        self.euler_config_listboxes = self.generate_euler_config_listboxes(width=default_width, height=default_height)

        self.run_euler_jobs_button = ttk.Button(self.root, text="Run euler jobs",
                                                command=self.execute_run_euler_jobs_script)
        self.run_euler_jobs_button.grid(row=self.current_ui_row, column=0, columnspan=3, pady=20)
        self.current_ui_row += 1

        self.run_euler_jobs_button = ttk.Button(self.root, text="Write out sample prompt",
                                                command=self.execute_write_out_sample_prompt)
        self.run_euler_jobs_button.grid(row=self.current_ui_row, column=0, columnspan=3, pady=20)
        self.current_ui_row += 1

        # Output label to show the script execution result
        self.output_label = ttk.Label(self.root, text="")
        self.output_label.grid(row=self.current_ui_row, column=0, columnspan=3, pady=5)
        self.current_ui_row += 1
        self.root.mainloop()

    def generate_euler_config_listboxes(self, width: int, height: int):
        config_listboxes = {}
        for field in EulerConfigValues.__annotations__:
            # Retrieve the list of possible values for the field
            values = EULER_CONFIG_POSSIBLE_VALUES.get(field, [])
            default_value = EULER_CONFIG_DEFAULT_VALUES.get(field, None)
            config_listboxes[field] = self.generate_list_box(label_text=field.replace('_', ' ').title() + ':',
                                                             select_mode=tk.SINGLE,
                                                             box_width=width,
                                                             box_height=height,
                                                             box_values=values,
                                                             default_value=default_value)
        return config_listboxes

    def generate_list_box(self, label_text: str, select_mode: str, box_width: int, box_height: int, box_values: List,
                          default_value: Any = None) -> tk.Listbox:
        box_label = ttk.Label(self.root, text=label_text)
        box_label.grid(row=self.current_ui_row, column=0, sticky='w', padx=5, pady=5)
        listbox = tk.Listbox(self.root, selectmode=select_mode, exportselection=0, width=box_width, height=box_height)
        listbox.grid(row=self.current_ui_row, column=1, padx=5, pady=5)
        self.current_ui_row += 1
        # Pre-fill the listbox with default values
        for value in box_values:
            listbox.insert(tk.END, value)
        if default_value is not None:
            self.set_default_value(listbox, box_values, default_value)
        return listbox

    @staticmethod
    def set_default_value(listbox, values, default_value):
        if default_value in values:
            default_index = values.index(default_value)
            listbox.selection_set(default_index)

    @staticmethod
    def list_files_by_task(root_dir):
        task_hierarchy = {}
        config_name_to_path_map = {}

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                full_path = os.path.join(root, file)
                parts = full_path.split(os.sep)

                # Find the directory part starting with 'task'
                task_dir = next((part for part in parts if part.startswith('task')), None)

                if not task_dir:
                    continue

                model_name = parts[2]
                num_shot = parts[4]
                fewshot_stategy = parts[5]
                seed = parts[6]
                prompt_version = parts[7].split(f"_{model_name}")[0]

                # Initialize dictionary structure if not present
                task_hierarchy.setdefault(task_dir, {}).setdefault(model_name, {})
                full_name = f"{prompt_version}_{num_shot}_{fewshot_stategy}_{seed}_{model_name}_{task_dir}"
                if full_name in task_hierarchy[task_dir][model_name]:
                    raise ValueError(f"Duplicate config: {full_path}")
                task_hierarchy[task_dir][model_name][full_name] = {"num_shot": num_shot,
                                                                   "fewshot_stategy": fewshot_stategy, "seed": seed,
                                                                   "full_path": full_path}
                config_name_to_path_map[full_name] = full_path

        return task_hierarchy, config_name_to_path_map

    def update_models_listbox(self, event):
        # Clear the current items from the dynamic_listbox
        self.models_listbox.delete(0, tk.END)
        tasks_names = [self.tasks_listbox.get(i) for i in self.tasks_listbox.curselection()]
        if len(tasks_names) == 0:
            return

        task_models = []
        for task_name in tasks_names:
            task_models.extend(list(self.task_hierarchy[task_name].keys()))

        sorted_model_names = sorted(task_models)
        for task_model in task_models:
            self.models_listbox.insert(tk.END, task_model)

    def update_configs_listbox(self, event):
        # Clear the current items from the dynamic_listbox
        self.configs_listbox.delete(0, tk.END)
        tasks_names = [self.tasks_listbox.get(i) for i in self.tasks_listbox.curselection()]
        if len(tasks_names) == 0:
            return

        models_names = [self.models_listbox.get(i) for i in self.models_listbox.curselection()]
        for task_name in tasks_names:
            task_models = list(self.task_hierarchy[task_name].keys())

            for task_model in task_models:
                if task_model not in models_names:
                    continue
                config_files = sorted(list(self.task_hierarchy[task_name][task_model].keys()))
                for config_file in config_files:
                    self.configs_listbox.insert(tk.END, config_file)

    @staticmethod
    def get_euler_code_path() -> str:
        # Read the existing JSON configuration
        with open(EULER_JSON_CONFIG_PATH, 'r') as file:
            config = json.load(file)
            return config["code_path"]

    def copy_prompt_template_to_euler(self, config_file_paths: List):
        euler_code_path = self.get_euler_code_path()
        prompt_template_paths = set()
        for config_file_path in config_file_paths:
            yaml_config = read_yaml(config_file_path)
            prompt_template_paths.add(yaml_config["task_configs"][0]["prompt_template"])
        for prompt_template_path in prompt_template_paths:
            scp_command = ["scp",
                           prompt_template_path,
                           f"euler:{euler_code_path}/{prompt_template_path}"]
            subprocess.run(scp_command, check=True)
            print(f"Copied prompt template to euler: {prompt_template_path}")

    def execute_run_euler_jobs_script(self):
        config_file_paths = [self.config_name_to_path_map[self.configs_listbox.get(i)] for i in
                             self.configs_listbox.curselection()]

        self.copy_prompt_template_to_euler(config_file_paths)

        self.update_euler_json_config()

        if len(config_file_paths) == 0:
            messagebox.showwarning("Warning", "Please select at least one config file!")
            return

        configs_str = " ".join(config_file_paths)
        # Call the bash script with the values as arguments
        call(['euler_scripts/euler_run_scheduler_v2.sh', configs_str])

    def execute_write_out_sample_prompt(self):
        config_file_paths = [self.config_name_to_path_map[self.configs_listbox.get(i)] for i in
                             self.configs_listbox.curselection()]
        if len(config_file_paths) == 0:
            messagebox.showwarning("Warning", "Please select at least one config file!")
            return
        command = "python3"
        script = "scripts/write_out.py"
        arg = "--config"
        for config_file in config_file_paths:
            cmd = [command, script, arg, config_file]
            subprocess.run(cmd)
            # call([f"python3 scripts/write_out.py --config {config_file}"])

    def get_selected_euler_config_values(self):
        # This method should return a dictionary of the selected values for each field
        selected_values = {}
        for field in EulerConfigValues.__annotations__:
            selected_index = self.euler_config_listboxes[field].curselection()
            if selected_index:
                selected_values[field] = self.euler_config_listboxes[field].get(selected_index[0])
        return selected_values

    def update_euler_json_config(self):
        # Read the existing JSON configuration
        with open(EULER_JSON_CONFIG_PATH, 'r') as file:
            config = json.load(file)

        # Get the selected EulerConfigValues
        selected_values = self.get_selected_euler_config_values()

        # Update the JSON configuration
        for euler_field, json_field in EULER_CONFIG_FIELD_MAPPING.items():
            if euler_field in selected_values:
                config[json_field] = selected_values[euler_field]

        # Write the updated configuration back to the JSON file
        with open(EULER_JSON_CONFIG_PATH, 'w') as file:
            json.dump(config, file, indent=4)

    def on_closing(self):
        # Any other cleanup tasks you might want to do can be added here
        self.root.destroy()  # This will close the application


def main():
    root_config_dir = "configs/faithfulness_benchmark_final_configs"
    EulerConfigRunUIManager(root_config_dir)


if __name__ == "__main__":
    main()
