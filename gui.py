import tkinter as tk
from tkinter import filedialog, messagebox

from gpx_overlay import (
    DEFAULT_CLIP_DURATION_SECONDS,
    DEFAULT_FPS,
    DEFAULT_FONT_PATH,
    DEFAULT_RESOLUTION,
    DEFAULT_ELEMENT_CONFIGS,
)
from gpx_overlay.gpx_parser import parse_gpx
from gpx_overlay.video_renderer import generate_gpx_video


def main() -> None:
    root = tk.Tk()
    root.title("GPX Overlay")

    gpx_path = tk.StringVar()
    output_path = tk.StringVar()
    total_duration = tk.IntVar(value=DEFAULT_CLIP_DURATION_SECONDS)

    def select_gpx() -> None:
        path = filedialog.askopenfilename(filetypes=[("GPX files", "*.gpx")])
        if path:
            gpx_path.set(path)
            _, start, end = parse_gpx(path)
            if start and end:
                duration = int((end - start).total_seconds())
                total_duration.set(duration)
                start_scale.config(to=duration)
                duration_scale.config(to=duration)
            else:
                messagebox.showerror("Erreur", "Fichier GPX invalide")

    def select_output() -> None:
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if path:
            output_path.set(path)

    def run() -> None:
        gpx = gpx_path.get()
        out = output_path.get()
        if not gpx or not out:
            messagebox.showerror("Erreur", "Choisissez les fichiers d'entrée et de sortie")
            return
        start_offset = start_scale.get()
        clip_duration = duration_scale.get()
        if start_offset + clip_duration > total_duration.get():
            messagebox.showerror("Erreur", "Durée sélectionnée trop longue")
            return
        ok = generate_gpx_video(
            gpx,
            out,
            start_offset,
            clip_duration,
            DEFAULT_FPS,
            DEFAULT_RESOLUTION,
            DEFAULT_FONT_PATH,
            DEFAULT_ELEMENT_CONFIGS,
        )
        if ok:
            messagebox.showinfo("Succès", "Vidéo générée")
        else:
            messagebox.showerror("Erreur", "Échec du rendu")

    tk.Button(root, text="Fichier GPX", command=select_gpx).grid(row=0, column=0, padx=5, pady=5)
    tk.Entry(root, textvariable=gpx_path, width=40).grid(row=0, column=1, padx=5, pady=5)
    tk.Button(root, text="Fichier sortie", command=select_output).grid(row=1, column=0, padx=5, pady=5)
    tk.Entry(root, textvariable=output_path, width=40).grid(row=1, column=1, padx=5, pady=5)

    tk.Label(root, text="Début (s)").grid(row=2, column=0)
    start_scale = tk.Scale(root, from_=0, to=total_duration.get(), orient="horizontal", length=300)
    start_scale.grid(row=2, column=1, padx=5, pady=5)

    tk.Label(root, text="Durée (s)").grid(row=3, column=0)
    duration_scale = tk.Scale(root, from_=1, to=total_duration.get(), orient="horizontal", length=300)
    duration_scale.grid(row=3, column=1, padx=5, pady=5)

    tk.Button(root, text="Générer", command=run).grid(row=4, column=0, columnspan=2, pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
