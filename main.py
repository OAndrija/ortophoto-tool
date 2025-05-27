import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image
from customtkinter import CTkImage
from tkinter import filedialog

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ImageViewerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Orthophoto Tool")
        self.geometry("1200x1000")
        self.resizable(False, False)

        self.app_state = "selecting_images"
        self.left_points = []
        self.right_points = []
        self.merged_image_array = None
        self.merged_points = []
        self.merged_labels = []
        self.real_world_points = []
        self.image_to_world_matrix = None
        self.world_x_map = None
        self.world_y_map = None
        self.left_image_array = None
        self.right_image_array = None

        self.instruction_label = ctk.CTkLabel(self, text="Load both images to begin.", font=ctk.CTkFont(size=16))
        self.instruction_label.pack(pady=(20, 5))

        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.pack(expand=True, fill="both", pady=20, padx=30)

        self.left_image_label = ctk.CTkLabel(self.image_frame, text="", anchor="nw", width=450, height=450)
        self.left_image_label.pack(side="left", padx=20, pady=10)
        self.left_image_label.bind("<Button-1>", self.on_left_image_click)

        self.right_image_label = ctk.CTkLabel(self.image_frame, text="", anchor="nw", width=450, height=450)
        self.right_image_label.pack(side="right", padx=20, pady=10)
        self.right_image_label.bind("<Button-1>", self.on_right_image_click)

        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(pady=(5, 15))

        self.load_left_button = ctk.CTkButton(self.button_frame, text="Load Left Image", command=self.load_left_image)
        self.load_left_button.grid(row=0, column=0, padx=20, pady=10)

        self.load_right_button = ctk.CTkButton(self.button_frame, text="Load Right Image", command=self.load_right_image)
        self.load_right_button.grid(row=0, column=1, padx=20, pady=10)

        self.done_button = ctk.CTkButton(self.button_frame, text="Done", command=self.on_done)
        self.done_button.grid(row=0, column=2, padx=20, pady=10)

        self.merge_button = ctk.CTkButton(self.button_frame, text="Merge Images", command=self.merge_images)
        self.merge_button.grid(row=0, column=3, padx=20, pady=10)
        self.merge_button.configure(state="disabled")

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

    def draw_points(self, image, points, labels=None):
        for idx, point in enumerate(points):
            cv2.circle(image, point, radius=3, color=(255, 0, 0), thickness=8)
            if labels and idx < len(labels):
                label = labels[idx]
                cv2.putText(image, label, (point[0]+10, point[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
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
                        self.merge_button.configure(state="normal")
                else:
                    self.update_instruction("❗ Please click within the RIGHT image bounds.")

    def merge_images(self):
        if len(self.left_points) == 2 and len(self.right_points) == 2:
            pts_left = np.array(self.left_points, dtype=np.float32)
            pts_right = np.array(self.right_points, dtype=np.float32)
            matrix = cv2.estimateAffinePartial2D(pts_right, pts_left, method=cv2.LMEDS)[0]
            if matrix is not None:
                h_left, w_left = self.left_image_array.shape[:2]
                h_right, w_right = self.right_image_array.shape[:2]
                corners = np.array([[0, 0], [w_right, 0], [w_right, h_right], [0, h_right]], dtype=np.float32)
                warped_corners = cv2.transform(np.array([corners]), matrix)[0]
                all_points = np.vstack((warped_corners, [[0, 0], [w_left, 0], [w_left, h_left], [0, h_left]]))
                [xmin, ymin] = np.floor(all_points.min(axis=0)).astype(int)
                [xmax, ymax] = np.ceil(all_points.max(axis=0)).astype(int)
                new_width = xmax - xmin
                new_height = ymax - ymin
                translation = np.array([[1, 0, -xmin], [0, 1, -ymin]])
                transform = translation @ np.vstack([matrix, [0, 0, 1]])
                warped_right = cv2.warpAffine(self.right_image_array, transform[:2], (new_width, new_height))
                canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                canvas[-ymin:h_left - ymin, -xmin:w_left - xmin] = self.left_image_array
                mask = (warped_right > 0).any(axis=2)
                canvas[mask] = warped_right[mask]
                self.merged_image_array = canvas
                self.merged_points = []
                self.merged_labels = []
                self.app_state = "selecting_points_merged"
                self.left_image_label.pack_forget()
                self.right_image_label.pack_forget()
                self.merged_image_label = ctk.CTkLabel(self.image_frame, text="", anchor="nw", width=900, height=600)
                self.merged_image_label.pack(expand=True, padx=20, pady=10)
                self.merged_image_label.bind("<Button-1>", self.on_merged_image_click)
                self.merged_image_label.bind("<Motion>", self.on_merged_image_hover)  # NEW
                self.update_instruction("Click 2 reference points on the MERGED image.")
                self.display_merged_image()
            else:
                self.update_instruction("❗ Could not calculate transformation matrix.")
        else:
            self.update_instruction("❗ Need exactly 2 points on each image.")

    def display_merged_image(self):
        image_copy = self.merged_image_array.copy()
        self.draw_points(image_copy, self.merged_points, self.merged_labels)
        merged_ctkimage = self.convert_to_ctkimage(image_copy)
        self.merged_image_label.configure(image=merged_ctkimage)
        self.merged_image_label.image = merged_ctkimage

    def on_merged_image_click(self, event):
        if self.app_state == "selecting_points_merged" and len(self.merged_points) < 2:
            if self.merged_image_array is not None:
                h, w = self.merged_image_array.shape[:2]
                x, y = event.x, event.y
                if 0 <= x < w and 0 <= y < h:
                    self.open_custom_input_dialog(x, y)
                else:
                    self.update_instruction("❗ Click within the merged image bounds.")

    def open_custom_input_dialog(self, x, y):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Enter Real-World Coordinates")
        dialog.geometry("300x200")
        dialog.grab_set()
        dialog.transient(self)

        ctk.CTkLabel(dialog, text=f"Real-world X for ({x}, {y}):").pack(pady=5)
        x_entry = ctk.CTkEntry(dialog)
        x_entry.pack(pady=5)

        ctk.CTkLabel(dialog, text=f"Real-world Y for ({x}, {y}):").pack(pady=5)
        y_entry = ctk.CTkEntry(dialog)
        y_entry.pack(pady=5)

        def submit():
            try:
                real_x = float(x_entry.get())
                real_y = float(y_entry.get())
            except ValueError:
                self.update_instruction("❗ Please enter valid numbers.")
                return

            self.merged_points.append((x, y))
            self.real_world_points.append((real_x, real_y))
            self.merged_labels.append(f"({real_x}, {real_y})")
            dialog.destroy()
            self.display_merged_image()
            self.update_instruction(f"Point {len(self.merged_points)} on MERGED selected.")

            if len(self.merged_points) == 2:
                self.app_state = "merged_points_done"
                self.update_instruction("✅ Reference points on merged image selected.")
                img_pts = np.array(self.merged_points, dtype=np.float32)
                world_pts = np.array(self.real_world_points, dtype=np.float32)
                self.image_to_world_matrix, _ = cv2.estimateAffinePartial2D(img_pts, world_pts)
                self.georeference_all_pixels()

        ctk.CTkButton(dialog, text="Submit", command=submit).pack(pady=10)

    def georeference_all_pixels(self):
        if self.image_to_world_matrix is not None:
            h, w = self.merged_image_array.shape[:2]
            x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
            pixel_coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1).astype(np.float32)
            ones = np.ones((pixel_coords.shape[0], 1), dtype=np.float32)
            pixel_coords_aug = np.hstack([pixel_coords, ones])
            world_coords = pixel_coords_aug @ self.image_to_world_matrix.T
            self.world_x_map = world_coords[:, 0].reshape(h, w)
            self.world_y_map = world_coords[:, 1].reshape(h, w)
            self.update_instruction("🌍 Georeferencing complete for all pixels.")

    def on_merged_image_hover(self, event):
        if self.world_x_map is not None and self.world_y_map is not None:
            h, w = self.merged_image_array.shape[:2]
            x, y = event.x, event.y
            if 0 <= x < w and 0 <= y < h:
                rx = self.world_x_map[y, x]
                ry = self.world_y_map[y, x]
                self.update_instruction(f"Hover: Image ({x}, {y}) → World ({rx:.2f}, {ry:.2f})")


if __name__ == "__main__":
    app = ImageViewerApp()
    app.mainloop()
