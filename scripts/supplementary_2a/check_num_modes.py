## IMPORTS
import os
import sys
sys.path.append(os.environ.get("LOCSCALE_2_SCRIPTS_PATH", ""))
sys.path.append("/home/abharadwaj1/dev/locscale")
import warnings
warnings.filterwarnings("ignore")
import csv
from tkinter import Tk, Button, Label, Entry, StringVar, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont

def collect_image_paths(root_dir):
    """
    Walk through the structured_data folder and collect all .png under model_free subfolders.
    Returns a list of (emdb_pdb, image_path).
    """
    paths = []
    for subdir, dirs, files in os.walk(root_dir):
        if os.path.basename(subdir) == 'model_free':
            # Look for .png in this folder or subfolders
            for dirpath, _, fnames in os.walk(subdir):
                for fname in fnames:
                    if fname.lower().endswith('.clusters_sigd.html.png'):
                        emdb_pdb = os.path.basename(os.path.dirname(subdir))
                        full_path = os.path.join(dirpath, fname)
                        paths.append((emdb_pdb, full_path))
    return sorted(paths)

class ModeAnnotator:
    def __init__(self, image_list, output_dir, csv_path):
        self.image_list = image_list
        self.output_dir = output_dir
        self.csv_path = csv_path
        self.index = 0
        self.responses = []  # List of dicts: {emdb, filename, num_modes}
        self.setup_ui()
        self.load_image()

    def setup_ui(self):
        self.root = Tk()
        self.root.title("Mode Annotator")

        self.img_label = Label(self.root)
        self.img_label.pack()

        self.prompt_var = StringVar()
        Label(self.root, textvariable=self.prompt_var).pack()
        self.prompt_var.set("How many modes are there in this image? (1-10)")

        self.entry = Entry(self.root)
        self.entry.pack()
        self.entry.bind('<Return>', lambda e: self.submit())

        btn_frame = Label(self.root)
        Button(btn_frame, text="Back", command=self.go_back).pack(side='left')
        Button(btn_frame, text="Next", command=self.submit).pack(side='left')
        btn_frame.pack()

    def load_image(self):
        emdb_pdb, path = self.image_list[self.index]
        img = Image.open(path)
        img.thumbnail((800, 600))
        self.current_image = img
        self.current_emdb = emdb_pdb
        self.current_fname = os.path.basename(path)
        self.tkimg = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tkimg)
        self.entry.delete(0, 'end')

    def submit(self):
        val = self.entry.get().strip()
        if not val.isdigit() or not (1 <= int(val) <= 10):
            self.prompt_var.set("Please enter an integer between 1 and 10.")
            return
        n = int(val)
        # record
        record = {
            'EMDB_ID': self.current_emdb,
            'filename': self.current_fname,
            'num_modes': n
        }
        # if editing existing
        if self.index < len(self.responses):
            self.responses[self.index] = record
        else:
            self.responses.append(record)
        # annotate and save image
        out_folder = os.path.join(self.output_dir, self.current_emdb)
        os.makedirs(out_folder, exist_ok=True)
        draw = ImageDraw.Draw(self.current_image)
        font = ImageFont.load_default()
        text = f"num_modes = {n}"
        draw.text((10, 10), text, font=font, fill=(255, 0, 0))
        save_path = os.path.join(out_folder, self.current_fname)
        self.current_image.save(save_path)
        # advance
        if self.index < len(self.image_list) - 1:
            self.index += 1
            self.load_image()
        else:
            self.finish()

    def go_back(self):
        if self.index > 0:
            self.index -= 1
            self.load_image()

    def finish(self):
        # write CSV
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['EMDB_ID', 'filename', 'num_modes'])
            writer.writeheader()
            writer.writerows(self.responses)
        # random-check prompt
        import random
        samples = random.sample(self.responses, min(5, len(self.responses)))
        # display each
        for rec in samples:
            emdb = rec['EMDB_ID']
            fname = rec['filename']
            n = rec['num_modes']
            path = os.path.join(self.output_dir, emdb, fname)
            img = Image.open(path)
            img.show()
            print(f"{emdb}/{fname} -> num_modes = {n}")
        self.root.quit()


def main():
    # PARAMS
    base_dir = "/home/abharadwaj1/papers/elife_paper/figure_information/archive_data/structured_data/supplementary_2a/bfactor_refinement_all_using_halfmaps"
    images = collect_image_paths(base_dir)
    output_dir = "/home/abharadwaj1/papers/elife_paper/figure_information/archive_data/structured_data/supplementary_2a/tobvalid_analysis/num_modes"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "mode_annotations.csv")

    annotator = ModeAnnotator(images, output_dir, csv_path)
    annotator.root.mainloop()

if __name__ == "__main__":
    main()
