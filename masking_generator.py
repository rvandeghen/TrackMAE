import numpy as np

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame
        self.visible_per_frame = self.num_patches_per_frame - self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}, visible per frame {}, mask per frame {}".format(
            self.total_patches, self.total_masks, self.num_patches_per_frame - self.num_masks_per_frame, self.num_masks_per_frame
        )
        return repr_str

    def __call__(self):
        if self.total_masks != 0:
            mask_per_frame = np.hstack([
                np.zeros(self.num_patches_per_frame - self.num_masks_per_frame, dtype=bool),
                np.ones(self.num_masks_per_frame, dtype=bool),
            ])
            np.random.shuffle(mask_per_frame)
            mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        else:
            mask = np.zeros(self.total_patches, dtype=bool)
        return mask


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio, frame_wise=False):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame
        self.frame_wise = frame_wise
        self.visible_per_frame = self.num_patches_per_frame - self.num_masks_per_frame
    
    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        if self.total_masks != 0:
            # Initialize an empty list to hold masks for each frame
            masks = []

            # Generate a random mask for each frame
            for _ in range(self.frames):
                mask_per_frame = np.hstack([
                    np.zeros(self.num_patches_per_frame - self.num_masks_per_frame, dtype=bool),
                    np.ones(self.num_masks_per_frame, dtype=bool),
                ])
                np.random.shuffle(mask_per_frame)
                masks.append(mask_per_frame)

            # Flatten the list of masks into a single array
            mask = np.array(masks).flatten()
        else:
            mask = np.zeros(self.total_patches, dtype=bool)
        return mask
