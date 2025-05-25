import customtkinter as ctk
import cv2
from PIL import Image
from customtkinter import CTkImage
from tkinter import filedialog

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ImageViewerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Orthophoto Tool")
        self.geometry("1000x700")
        self.resizable(False, False)

        self.app_state = "selecting_images"
        self.left_points = []
        self.right_points = []

        self.left_image_array = None
        self.right_image_array = None

        # Top row - Instructions
        self.instruction_label = ctk.CTkLabel(self, text="Load both images to begin.", font=ctk.CTkFont(size=16))
        self.instruction_label.pack(pady=(20, 5))

        # Middle row - Images
        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.pack(expand=True, fill="both", pady=20, padx=30)

        self.left_image_label = ctk.CTkLabel(self.image_frame, text="", anchor="nw", width=450, height=450)
        self.left_image_label.pack(side="left", padx=20, pady=10)
        self.left_image_label.bind("<Button-1>", self.on_left_image_click)

        self.right_image_label = ctk.CTkLabel(self.image_frame, text="", anchor="nw", width=450, height=450)
        self.right_image_label.pack(side="right", padx=20, pady=10)
        self.right_image_label.bind("<Button-1>", self.on_right_image_click)

        # Bottom row - Buttons
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(pady=(5, 15))

        self.load_left_button = ctk.CTkButton(self.button_frame, text="Load Left Image", command=self.load_left_image)
        self.load_left_button.grid(row=0, column=0, padx=20, pady=10)

        self.load_right_button = ctk.CTkButton(self.button_frame, text="Load Right Image", command=self.load_right_image)
        self.load_right_button.grid(row=0, column=1, padx=20, pady=10)

        self.done_button = ctk.CTkButton(self.button_frame, text="Done", command=self.on_done)
        self.done_button.grid(row=0, column=2, padx=20, pady=10)

    def update_instruction(self, text):
        self.instruction_label.configure(text=text)

    def load_left_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.tif")])
        if filepath:
            self.left_image_array = self.read_image(filepath)
            self.display_left_image()

    def load_right_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.tif")])
        if filepath:
            self.right_image_array = self.read_image(filepath)
            self.display_right_image()

    def on_done(self):
        if self.left_image_array is not None and self.right_image_array is not None:
            self.app_state = "selecting_points_left"
            self.load_left_button.grid_forget()
            self.load_right_button.grid_forget()
            self.done_button.configure(state="disabled")
            self.update_instruction("Click 2 tie points on the LEFT image.")
        else:
            self.update_instruction("❗ Load both images before continuing.")

    def read_image(self, path, max_size=(450, 450)):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        scale = min(max_size[0] / w, max_size[1] / h)
        image_resized = cv2.resize(image, (int(w * scale), int(h * scale)))
        return image_resized

    def convert_to_ctkimage(self, image_array):
        pil_image = Image.fromarray(image_array)
        return CTkImage(light_image=pil_image, size=pil_image.size)

    def draw_points(self, image, points):
        for point in points:
            cv2.circle(image, point, radius=3, color=(255, 0, 0), thickness=8)
        return image

    def display_left_image(self):
        image_copy = self.left_image_array.copy()
        self.draw_points(image_copy, self.left_points)
        self.left_image = self.convert_to_ctkimage(image_copy)
        self.left_image_label.configure(image=self.left_image)

    def display_right_image(self):
        image_copy = self.right_image_array.copy()
        self.draw_points(image_copy, self.right_points)
        self.right_image = self.convert_to_ctkimage(image_copy)
        self.right_image_label.configure(image=self.right_image)

    def on_left_image_click(self, event):
        if self.app_state == "selecting_points_left" and len(self.left_points) < 2:
            if self.left_image_array is not None:
                h, w = self.left_image_array.shape[:2]
                x, y = event.x, event.y
                if 0 <= x < w and 0 <= y < h:
                    self.left_points.append((x, y))
                    self.update_instruction(f"Selected point {len(self.left_points)} on LEFT: ({x}, {y})")
                    self.display_left_image()
                    if len(self.left_points) == 2:
                        self.app_state = "selecting_points_right"
                        self.update_instruction("Now click 2 tie points on the RIGHT image.")
                else:
                    self.update_instruction("❗ Please click within the LEFT image bounds.")


    def on_right_image_click(self, event):
        if self.app_state == "selecting_points_right" and len(self.right_points) < 2:
            if self.right_image_array is not None:
                h, w = self.right_image_array.shape[:2]
                x, y = event.x, event.y
                if 0 <= x < w and 0 <= y < h:
                    self.right_points.append((x, y))
                    self.update_instruction(f"Selected point {len(self.right_points)} on RIGHT: ({x}, {y})")
                    self.display_right_image()
                    if len(self.right_points) == 2:
                        self.app_state = "tie_points_done"
                        self.update_instruction("✅ Tie point selection complete. Ready for stitching.")
                else:
                    self.update_instruction("❗ Please click within the RIGHT image bounds.")



if __name__ == "__main__":
    app = ImageViewerApp()
    app.mainloop()
