import tkinter as tk
from tkinter import filedialog, messagebox
import rasterio
import numpy as np
from PIL import Image, ImageTk


class TIFFViewer:

    def __init__(self, root):

        self.root = root
        self.root.title("TIFF Viewer")
        self.root.geometry("1200x800")

        self.image = None
        self.photo = None

        # -------------------------------
        # Top bar
        # -------------------------------

        top = tk.Frame(root)
        top.pack(fill="x", padx=10, pady=10)

        tk.Button(
            top,
            text="Open TIFF",
            command=self.open_tiff,
            font=("Arial", 12)
        ).pack(side="left")

        self.info = tk.Label(
            top,
            text="No TIFF loaded",
            font=("Arial", 11)
        )

        self.info.pack(
            side="left",
            padx=20
        )

        # -------------------------------
        # Band selector
        # -------------------------------

        self.band_var = tk.IntVar(value=1)

        self.band_label = tk.Label(
            top,
            text="Band:"
        )

        self.band_label.pack(
            side="left",
            padx=(20, 5)
        )

        self.band_spin = tk.Spinbox(
            top,
            from_=1,
            to=1,
            textvariable=self.band_var,
            width=5,
            command=self.show_band
        )

        self.band_spin.pack(side="left")

        tk.Button(
            top,
            text="Show Band",
            command=self.show_band
        ).pack(
            side="left",
            padx=5
        )

        # -------------------------------
        # Image area
        # -------------------------------

        self.canvas = tk.Canvas(
            root,
            bg="black"
        )

        self.canvas.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

    # =====================================================
    # OPEN TIFF
    # =====================================================

    def open_tiff(self):

        path = filedialog.askopenfilename(
            title="Select TIFF",
            filetypes=[
                ("TIFF files", "*.tif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        try:

            self.dataset = rasterio.open(path)

            self.info.config(
                text=(
                    f"{self.dataset.width} × "
                    f"{self.dataset.height} | "
                    f"Bands: {self.dataset.count}"
                )
            )

            self.band_spin.config(
                to=self.dataset.count
            )

            self.band_var.set(1)

            self.show_band()

        except Exception as e:

            messagebox.showerror(
                "Error",
                f"Could not open TIFF:\n\n{e}"
            )

    # =====================================================
    # SHOW BAND
    # =====================================================

    def show_band(self):

        if not hasattr(self, "dataset"):
            return

        try:

            band_number = self.band_var.get()

            data = self.dataset.read(
                band_number
            ).astype(np.float32)

            data = np.nan_to_num(
                data,
                nan=0,
                posinf=0,
                neginf=0
            )

            # Percentile stretch
            low = np.percentile(
                data,
                2
            )

            high = np.percentile(
                data,
                98
            )

            if high > low:

                data = (
                    data - low
                ) / (
                    high - low
                )

            data = np.clip(
                data,
                0,
                1
            )

            data = (
                data * 255
            ).astype(np.uint8)

            image = Image.fromarray(
                data
            )

            # Fit image inside window
            max_width = 1100
            max_height = 650

            image.thumbnail(
                (
                    max_width,
                    max_height
                )
            )

            self.photo = ImageTk.PhotoImage(
                image
            )

            self.canvas.delete(
                "all"
            )

            self.canvas.create_image(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                image=self.photo,
                anchor="center"
            )

        except Exception as e:

            messagebox.showerror(
                "Error",
                str(e)
            )


# =========================================================
# RUN
# =========================================================

root = tk.Tk()

app = TIFFViewer(root)

root.mainloop()