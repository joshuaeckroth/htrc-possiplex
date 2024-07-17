import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import plotly.express as px
import umap.umap_ as umap
from scipy.special import fig


#Idea. We can have 2 buttons, one of them will ask to provide the original text of the book book (previously refactored)
# to be shown on a plot with the bright RED dot.
#The second button would ask to provide data from csv file to plot other dots which represent variations
#of the text.
def select_file_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        folder_entry.delete(0, tk.END)
        folder_entry.insert(0, file_path)

def save_results():
    file_path = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML files", "*.html")])
    if file_path:
        fig.write_html(file_path)
        messagebox.showinfo("Save", "File successfully saved!")

def read_csv_file(file_path):
    try:
        # Read the CSV file as raw text
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = f.readlines()

        # issues with quotations and delimiters (process each line separately for this reason)
        data = []
        for line in raw_data:
            # any surrounding quotes? Delete it. Also split by tab delimiter
            cleaned_line = line.strip().strip('"').split('\t')
            data.append([float(item.strip().strip('"')) for item in cleaned_line])

        embeddings_df = pd.DataFrame(data)
        return embeddings_df
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while reading the file: {e}")
        return pd.DataFrame()

def start_processing():
    file_path = folder_entry.get()
    if not file_path:
        messagebox.showwarning("Input required", "Please select a CSV file to analyze.")
        return

    try:
        embeddings_df = read_csv_file(file_path)

        # Dataframe empty?
        if embeddings_df.empty:
            messagebox.showerror("Error", "The selected file is empty or not in the correct format.")
            return

        # using UMAP to reduce dimensionality
        reducer = umap.UMAP(n_components=3, random_state=42)
        reduced_embeddings_3d = reducer.fit_transform(embeddings_df)

        # using PLOTLY to plot
        fig = px.scatter_3d(
            x=reduced_embeddings_3d[:, 0],
            y=reduced_embeddings_3d[:, 1],
            z=reduced_embeddings_3d[:, 2],
            title="3D UMAP for text variations 😺",
            labels={"x": "DIMENSION X", "y": "Dimension Y", "z": "Dimension Z"},
            opacity=0.7,  # Adjust marker opacity
            size_max=10  # Maximum size of the marker
        )
        # Adjusting marker size and adding more visibility
        fig.update_traces(marker=dict(size=5, opacity=0.4))

        fig.show()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


app = tk.Tk()
app.title("Embedding Visualization")

folder_label = tk.Label(app, text="Choose a CSV file to analyze:")
folder_label.pack()

folder_entry = tk.Entry(app, width=50)
folder_entry.pack()

browse_button = tk.Button(app, text="Choose the file", command=select_file_csv)
browse_button.pack()

process_button = tk.Button(app, text="Process the data", command=start_processing)
process_button.pack()

save_button = tk.Button(app, text="Save Results", command=save_results)
save_button.pack()

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(app, variable=progress_var, maximum=100)
progress_bar.pack()

app.mainloop()
