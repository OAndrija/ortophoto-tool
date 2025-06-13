import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image
from customtkinter import CTkImage
from tkinter import filedialog
from tkfontawesome import icon_to_image

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class OrthophotoTool(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Orthophoto Tool")
        self.geometry("1200x900")
        self.resizable(False, False)

        self.font_regular = ctk.CTkFont(family="Roboto", size=16)
        self.font_bold = ctk.CTkFont(family="Roboto", size=16, weight="bold")
        
        self.icon_load_left = icon_to_image("image", fill="white", scale_to_width=20)
        self.icon_load_right = icon_to_image("image", fill="white", scale_to_width=20)
        self.icon_done = icon_to_image("check", fill="white", scale_to_width=20)
        self.icon_merge = icon_to_image("object-group", fill="white", scale_to_width=20)
        self.icon_save = icon_to_image("save", fill="white", scale_to_width=20)
        self.icon_reset = icon_to_image("redo", fill="white", scale_to_width=20)

        button_style = {
            "height": 50,
            "font": self.font_regular,
            "corner_radius": 10,
            "compound": "left"
        }

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

        self.instruction_label = ctk.CTkLabel(
            self,
            text="Load both images to begin.",
            font=self.font_regular
        )
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

        self.load_left_button = ctk.CTkButton(
            self.button_frame, text=" Load Left Image", image=self.icon_load_left, command=self.load_left_image, **button_style
        )
        self.load_left_button.grid(row=0, column=0, padx=20, pady=10)

        self.load_right_button = ctk.CTkButton(
            self.button_frame, text=" Load Right Image", image=self.icon_load_right, command=self.load_right_image, **button_style
        )
        self.load_right_button.grid(row=0, column=1, padx=20, pady=10)

        self.done_button = ctk.CTkButton(
            self.button_frame, text=" Done", image=self.icon_done, command=self.on_done, **button_style
        )
        self.done_button.grid(row=0, column=2, padx=20, pady=10)
        self.done_button.grid_remove()

        self.merge_button = ctk.CTkButton(
            self.button_frame, text=" Merge Images", image=self.icon_merge, command=self.merge_images, **button_style
        )
        self.merge_button.grid(row=0, column=3, padx=20, pady=10)
        self.merge_button.grid_remove()

        self.save_button = ctk.CTkButton(
            self.button_frame, text=" Save Image", image=self.icon_save, command=self.save_image, **button_style
        )
        self.save_button.grid(row=0, column=4, padx=20, pady=10)
        self.save_button.grid_remove()

        self.reset_button = ctk.CTkButton(
            self.button_frame, text=" Reset", image=self.icon_reset, command=self.reset_app, **button_style
        )
        self.reset_button.grid(row=0, column=5, padx=20, pady=10)
        self.reset_button.grid_remove()



    def update_instruction(self, text):
        self.instruction_label.configure(text=text)

    def load_left_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.tif")])
        if filepath:
            self.left_image_array = self.read_image(filepath)
            self.display_left_image()
            self.check_show_done_button()

    def load_right_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.tif")])
        if filepath:
            self.right_image_array = self.read_image(filepath)
            self.display_right_image()
            self.check_show_done_button()

    def check_show_done_button(self):
        if self.left_image_array is not None and self.right_image_array is not None:
            self.done_button.grid()
            self.update_instruction("Click Done to proceed.")

    def on_done(self):
        self.app_state = "selecting_points_left"
        self.load_left_button.grid_remove()
        self.load_right_button.grid_remove()
        self.done_button.grid_remove()
        self.update_instruction("Click 2 tie points on the LEFT image.")

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
            x, y = event.x, event.y
            if 0 <= x < self.left_image_array.shape[1] and 0 <= y < self.left_image_array.shape[0]:
                self.left_points.append((x, y))
                self.update_instruction(f"Selected point {len(self.left_points)} on LEFT: ({x}, {y})")
                self.display_left_image()
                if len(self.left_points) == 2:
                    self.app_state = "selecting_points_right"
                    self.update_instruction("Now click 2 tie points on the RIGHT image.")

    def on_right_image_click(self, event):
        if self.app_state == "selecting_points_right" and len(self.right_points) < 2:
            x, y = event.x, event.y
            if 0 <= x < self.right_image_array.shape[1] and 0 <= y < self.right_image_array.shape[0]:
                self.right_points.append((x, y))
                self.update_instruction(f"Selected point {len(self.right_points)} on RIGHT: ({x}, {y})")
                self.display_right_image()
                if len(self.right_points) == 2:
                    self.app_state = "tie_points_done"
                    self.update_instruction("✅ Tie point selection complete. Ready for stitching.")
                    self.merge_button.grid()

    def merge_images(self):
        if len(self.left_points) == 2 and len(self.right_points) == 2:
            
            #Mapira desne tačke na leve (Rotacija, skaliranje i translacija)
            matrix = cv2.estimateAffinePartial2D(np.array(self.right_points), np.array(self.left_points), method=cv2.LMEDS)[0]
            
            if matrix is not None:
                #Dimenzije dve pocetne slike
                h_left, w_left = self.left_image_array.shape[:2]
                h_right, w_right = self.right_image_array.shape[:2]
                
                #Koordinate uglova desne slike
                corners = np.array([[0, 0], [w_right, 0], [w_right, h_right], [0, h_right]], dtype=np.float32)
                
                #Transformacija uglova desne slike pomoću afine matrice
                warped_corners = cv2.transform(np.array([corners]), matrix)[0]
                
                #Kombinovanje svih uglova (transformisanih desne i originalnih leve slike)
                all_points = np.vstack((warped_corners, [[0, 0], [w_left, 0], [w_left, h_left], [0, h_left]]))
                
                #Izračunavanje minimalnih i maksimalnih koordinata nove slike
                [xmin, ymin] = np.floor(all_points.min(axis=0)).astype(int)
                [xmax, ymax] = np.ceil(all_points.max(axis=0)).astype(int)
                
                #Širina i visina nove slike
                new_width = xmax - xmin
                new_height = ymax - ymin
                
                #Primenjujemo translaciju da bismo izbegli negativne koordinate
                translation = np.array([[1, 0, -xmin], [0, 1, -ymin]])
                
                #Kombinujemo translaciju sa afinom matricom
                transform = translation @ np.vstack([matrix, [0, 0, 1]])
                
                #Warpujemo desnu sliku
                warped_right = cv2.warpAffine(self.right_image_array, transform[:2], (new_width, new_height))
                
                canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                
                #Ubacujemo levu sliku na odgovarajuće mesto na platnu
                canvas[-ymin:h_left - ymin, -xmin:w_left - xmin] = self.left_image_array
                
                #Maskiranje svih piksela gde desna slika ima podatke (nije crna)
                mask = (warped_right > 0).any(axis=2)
                
                #Ubacujemo desnu sliku preko platna
                canvas[mask] = warped_right[mask]
                
                self.merged_image_array = canvas
                self.app_state = "selecting_points_merged"
                self.left_image_label.pack_forget()
                self.right_image_label.pack_forget()
                self.merged_image_label = ctk.CTkLabel(self.image_frame, text="", anchor="nw", width=900, height=600)
                self.merged_image_label.pack(expand=True, padx=20, pady=10)
                self.merged_image_label.bind("<Button-1>", self.on_merged_image_click)
                self.merged_image_label.bind("<Motion>", self.on_merged_image_hover)
                self.display_merged_image()
                self.update_instruction("Click 2 reference points on the MERGED image.")
                self.merge_button.grid_remove()

    def display_merged_image(self):
        image_copy = self.merged_image_array.copy()
        self.draw_points(image_copy, self.merged_points, self.merged_labels)
        merged_ctkimage = self.convert_to_ctkimage(image_copy)
        self.merged_image_label.configure(image=merged_ctkimage)
        self.merged_image_label.image = merged_ctkimage

    def on_merged_image_click(self, event):
        if self.app_state == "selecting_points_merged" and len(self.merged_points) < 2:
            x, y = event.x, event.y
            if 0 <= x < self.merged_image_array.shape[1] and 0 <= y < self.merged_image_array.shape[0]:
                self.open_custom_input_dialog(x, y)

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
                self.image_to_world_matrix, _ = cv2.estimateAffinePartial2D(np.array(self.merged_points), np.array(self.real_world_points))
                self.georeference_all_pixels()
                self.save_button.grid()
                self.reset_button.grid()

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
            x, y = event.x, event.y
            if 0 <= x < self.world_x_map.shape[1] and 0 <= y < self.world_x_map.shape[0]:
                rx = self.world_x_map[y, x]
                ry = self.world_y_map[y, x]
                self.update_instruction(f"Hover: Image ({x}, {y}) → World ({rx:.2f}, {ry:.2f})")

    def save_image(self):
        if self.merged_image_array is not None:
            filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                                    filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
            if filepath:
                image_rgb = cv2.cvtColor(self.merged_image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, image_rgb)
                self.update_instruction(f"✅ Image saved to {filepath}")

    def reset_app(self):
        self.app_state = "selecting_images"
        self.left_points.clear()
        self.right_points.clear()
        self.merged_image_array = None
        self.merged_points.clear()
        self.merged_labels.clear()
        self.real_world_points.clear()
        self.image_to_world_matrix = None
        self.world_x_map = None
        self.world_y_map = None
        self.left_image_array = None
        self.right_image_array = None

        if hasattr(self, 'merged_image_label'):
            self.merged_image_label.destroy()
            del self.merged_image_label

        self.left_image_label.pack(side="left", padx=20, pady=10)
        self.right_image_label.pack(side="right", padx=20, pady=10)
        self.left_image_label.configure(image="", text="")
        self.right_image_label.configure(image="", text="")

        self.load_left_button.grid()
        self.load_right_button.grid()
        self.done_button.grid_remove()
        self.merge_button.grid_remove()
        self.save_button.grid_remove()
        self.reset_button.grid_remove()

        self.update_instruction("Load both images to begin.")



if __name__ == "__main__":
    app = OrthophotoTool()
    app.mainloop()
