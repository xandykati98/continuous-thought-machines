import numpy as np
import cv2
import torch
import os
import imageio
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patheffects
from matplotlib.patches import Rectangle
mpl.use('Agg')
import seaborn as sns
import numpy as np
from tqdm.auto import tqdm
sns.set_style('darkgrid')

from tqdm.auto import tqdm
from scipy import ndimage
import umap
from scipy.special import softmax
from scipy.stats import pearsonr
import json

import subprocess as sp
import cv2 # Still potentially useful for color conversion checks if needed
import os

def save_frames_to_mp4(frames, output_filename, fps=15.0, gop_size=None, crf=23, preset='medium', pix_fmt='yuv420p'):
    """
    Saves a list of NumPy array frames to an MP4 video file using FFmpeg via subprocess.

    Includes fix for odd frame dimensions by padding to the nearest even number using -vf pad.

    Requires FFmpeg to be installed and available in the system PATH.

    Args:
        frames (list): A list of NumPy arrays representing the video frames.
                       Expected format: uint8, (height, width, 3) for BGR color
                       or (height, width) for grayscale. Should be consistent.
        output_filename (str): The path and name for the output MP4 file.
        fps (float, optional): Frames per second for the output video. Defaults to 15.0.
        gop_size (int, optional): Group of Pictures (GOP) size. This determines the
                                  maximum interval between keyframes. Lower values
                                  mean more frequent keyframes (better seeking, larger file).
                                  Defaults to int(fps) (approx 1 keyframe per second).
        crf (int, optional): Constant Rate Factor for H.264 encoding. Lower values mean
                             better quality and larger files. Typical range: 18-28.
                             Defaults to 23.
        preset (str, optional): FFmpeg encoding speed preset. Affects encoding time
                                and compression efficiency. Options include 'ultrafast',
                                'superfast', 'veryfast', 'faster', 'fast', 'medium',
                                'slow', 'slower', 'veryslow'. Defaults to 'medium'.
    """
    if not frames:
        print("Error: The 'frames' list is empty. No video to save.")
        return

    # --- Determine Parameters from First Frame ---
    try:
        first_frame = frames[0]
        print(first_frame.shape)
        if not isinstance(first_frame, np.ndarray):
             print(f"Error: Frame 0 is not a NumPy array (type: {type(first_frame)}).")
             return

        frame_height, frame_width = first_frame.shape[:2]
        frame_size_str = f"{frame_width}x{frame_height}"

        # Determine input pixel format based on first frame's shape
        if len(first_frame.shape) == 3 and first_frame.shape[2] == 3:
            input_pixel_format = 'bgr24' # Assume OpenCV's default BGR uint8
            expected_dims = 3
            print(f"Info: Detected color frames (shape: {first_frame.shape}). Expecting BGR input.")
        elif len(first_frame.shape) == 2:
            input_pixel_format = 'gray'
            expected_dims = 2
            print(f"Info: Detected grayscale frames (shape: {first_frame.shape}).")
        else:
            print(f"Error: Unsupported frame shape {first_frame.shape}. Must be (h, w) or (h, w, 3).")
            return

        if first_frame.dtype != np.uint8:
             print(f"Warning: First frame dtype is {first_frame.dtype}. Will attempt conversion to uint8.")

    except IndexError:
        print("Error: Could not access the first frame to determine dimensions.")
        return
    except Exception as e:
         print(f"Error processing first frame: {e}")
         return

    # --- Set GOP size default if not provided ---
    if gop_size is None:
        gop_size = int(fps)
        print(f"Info: GOP size not specified, defaulting to {gop_size} (approx 1 keyframe/sec).")

    # --- Construct FFmpeg Command ---
    # ADDED -vf pad filter to ensure even dimensions for libx264/yuv420p
    # It calculates the nearest even dimensions >= original dimensions
    # Example: 1600x1351 -> 1600x1352
    pad_filter = "pad=ceil(iw/2)*2:ceil(ih/2)*2"

    command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', input_pixel_format,
        '-s', frame_size_str,
        '-r', str(float(fps)),
        '-i', '-',
        '-vf', pad_filter, # <--- ADDED VIDEO FILTER HERE
        '-c:v', 'libx264',
        '-pix_fmt', pix_fmt,
        '-preset', preset,
        '-crf', str(crf),
        '-g', str(gop_size),
        '-movflags', '+faststart',
        output_filename
    ]

    print(f"\n--- Starting FFmpeg ---")
    print(f"Output File: {output_filename}")
    print(f"Parameters: FPS={fps}, Size={frame_size_str}, GOP={gop_size}, CRF={crf}, Preset={preset}")
    print(f"Applying Filter: -vf {pad_filter} (Ensures even dimensions)")
    # print(f"FFmpeg Command: {' '.join(command)}") # Uncomment for debugging

    # --- Execute FFmpeg via Subprocess ---
    try:
        process = sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)

        print(f"\nWriting {len(frames)} frames to FFmpeg...")
        progress_interval = max(1, len(frames) // 10) # Print progress roughly 10 times

        for i, frame in enumerate(frames):
            # Basic validation and conversion for each frame
            if not isinstance(frame, np.ndarray):
                 print(f"Warning: Frame {i} is not a numpy array (type: {type(frame)}). Skipping.")
                 continue
            if frame.shape[0] != frame_height or frame.shape[1] != frame_width:
                print(f"Warning: Frame {i} has different dimensions {frame.shape[:2]}! Expected ({frame_height},{frame_width}). Skipping.")
                continue

            current_dims = len(frame.shape)
            if current_dims != expected_dims:
                 print(f"Warning: Frame {i} has inconsistent dimensions ({current_dims}D vs expected {expected_dims}D). Skipping.")
                 continue
            if expected_dims == 3 and frame.shape[2] != 3:
                 print(f"Warning: Frame {i} is color but doesn't have 3 channels ({frame.shape}). Skipping.")
                 continue

            if frame.dtype != np.uint8:
                try:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                except Exception as clip_err:
                     print(f"Error clipping/converting frame {i} dtype: {clip_err}. Skipping.")
                     continue

            # Write frame bytes to FFmpeg's stdin
            try:
                 process.stdin.write(frame.tobytes())
            except (OSError, BrokenPipeError) as pipe_err:
                 print(f"\nError writing frame {i} to FFmpeg stdin: {pipe_err}")
                 print("FFmpeg process likely terminated prematurely. Check FFmpeg errors below.")
                 try:
                     # Immediately try to read stderr if pipe breaks
                     stderr_output_on_error = process.stderr.read()
                     if stderr_output_on_error:
                          print("\n--- FFmpeg stderr output on error ---")
                          print(stderr_output_on_error.decode(errors='ignore'))
                          print("--- End FFmpeg stderr ---")
                 except Exception as read_err:
                     print(f"(Could not read stderr after pipe error: {read_err})")
                 return
            except Exception as write_err:
                 print(f"Unexpected error writing frame {i}: {write_err}. Skipping.")
                 continue

            if (i + 1) % progress_interval == 0 or (i + 1) == len(frames):
                 print(f"  Processed frame {i + 1}/{len(frames)}")

        print("\nFinished writing frames. Closing FFmpeg stdin and waiting for completion...")
        process.stdin.close()
        stdout, stderr = process.communicate()
        return_code = process.wait()

        print("\n--- FFmpeg Final Status ---")
        if return_code == 0:
            print(f"FFmpeg process completed successfully.")
            print(f"Video saved as: {output_filename}")
        else:
            print(f"FFmpeg process failed with return code {return_code}.")
            print("--- FFmpeg Standard Error Output: ---")
            print(stderr.decode(errors='replace')) # Print stderr captured by communicate()
            print("--- End FFmpeg Output ---")
            print("Review the FFmpeg error message above for details (e.g., dimension errors, parameter issues).")

    except FileNotFoundError:
        print("\n--- FATAL ERROR ---")
        print("Error: 'ffmpeg' command not found.")
        print("Please ensure FFmpeg is installed and its directory is included in your system's PATH environment variable.")
        print("Download from: https://ffmpeg.org/")
        print("-------------------")
    except Exception as e:
        print(f"\nAn unexpected error occurred during FFmpeg execution: {e}")

def find_island_centers(array_2d, threshold):
    """
    Finds the center of mass of each island (connected component) in a 2D array.

    Args:
        array_2d: A 2D numpy array of values.
        threshold: The threshold to binarize the array.

    Returns:
        A list of tuples (y, x) representing the center of mass of each island.
    """
    binary_image = array_2d > threshold
    labeled_image, num_labels = ndimage.label(binary_image)
    centers = []
    areas = []  # Store the area of each island
    for i in range(1, num_labels + 1):
        island = (labeled_image == i)
        total_mass = np.sum(array_2d[island])
        if total_mass > 0:
            y_coords, x_coords = np.mgrid[:array_2d.shape[0], :array_2d.shape[1]]
            x_center = np.average(x_coords[island], weights=array_2d[island])
            y_center = np.average(y_coords[island], weights=array_2d[island])
            centers.append((round(y_center, 4), round(x_center, 4)))
            areas.append(np.sum(island))  # Calculate area of the island
    return centers, areas

def plot_neural_dynamics(post_activations_history, N_to_plot, save_location, axis_snap=False, N_per_row=5, which_neurons_mid=None, mid_colours=None, use_most_active_neurons=False):
    assert N_to_plot%N_per_row==0, f'For nice visualisation, N_to_plot={N_to_plot} must be a multiple of N_per_row={N_per_row}'
    assert post_activations_history.shape[-1] >= N_to_plot
    figscale = 2
    aspect_ratio = 3
    mosaic = np.array([[f'{i}'] for i in range(N_to_plot)]).flatten().reshape(-1, N_per_row)
    fig_synch, axes_synch = plt.subplot_mosaic(mosaic=mosaic, figsize=(figscale*mosaic.shape[1]*aspect_ratio*0.2, figscale*mosaic.shape[0]*0.2))
    fig_mid, axes_mid = plt.subplot_mosaic(mosaic=mosaic, figsize=(figscale*mosaic.shape[1]*aspect_ratio*0.2, figscale*mosaic.shape[0]*0.2), dpi=200)

    palette = sns.color_palette("husl", 8)
    
    which_neurons_synch = np.arange(N_to_plot)
    # which_neurons_mid = np.arange(N_to_plot, N_to_plot*2) if post_activations_history.shape[-1] >= 2*N_to_plot else np.random.choice(np.arange(post_activations_history.shape[-1]), size=N_to_plot, replace=True)
    random_indices = np.random.choice(np.arange(post_activations_history.shape[-1]), size=N_to_plot, replace=post_activations_history.shape[-1] < N_to_plot)
    if use_most_active_neurons:
        metric = np.abs(np.fft.rfft(post_activations_history, axis=0))[3:].mean(0).std(0)
        random_indices = np.argsort(metric)[-N_to_plot:]
        np.random.shuffle(random_indices)
    which_neurons_mid = which_neurons_mid if which_neurons_mid is not None else random_indices

    if mid_colours is None:
        mid_colours = [palette[np.random.randint(0, 8)] for ndx in range(N_to_plot)]
    with tqdm(total=N_to_plot, initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:
        pbar_inner.set_description('Plotting neural dynamics')
        for ndx in range(N_to_plot):
            
            ax_s = axes_synch[f'{ndx}']
            ax_m = axes_mid[f'{ndx}']

            traces_s = post_activations_history[:,:,which_neurons_synch[ndx]].T
            traces_m = post_activations_history[:,:,which_neurons_mid[ndx]].T
            c_s = palette[np.random.randint(0, 8)]
            c_m = mid_colours[ndx]

            for traces_s_here, traces_m_here in zip(traces_s, traces_m):
                ax_s.plot(np.arange(len(traces_s_here)), traces_s_here, linestyle='-', color=c_s, alpha=0.05, linewidth=0.6)
                ax_m.plot(np.arange(len(traces_m_here)), traces_m_here, linestyle='-', color=c_m, alpha=0.05, linewidth=0.6)

            
            ax_s.plot(np.arange(len(traces_s[0])), traces_s[0], linestyle='-', color='white', alpha=1, linewidth=2.5)
            ax_s.plot(np.arange(len(traces_s[0])), traces_s[0], linestyle='-', color=c_s, alpha=1, linewidth=1.3)
            ax_s.plot(np.arange(len(traces_s[0])), traces_s[0], linestyle='-', color='black', alpha=1, linewidth=0.3)
            ax_m.plot(np.arange(len(traces_m[0])), traces_m[0], linestyle='-', color='white', alpha=1, linewidth=2.5)
            ax_m.plot(np.arange(len(traces_m[0])), traces_m[0], linestyle='-', color=c_m, alpha=1, linewidth=1.3)
            ax_m.plot(np.arange(len(traces_m[0])), traces_m[0], linestyle='-', color='black', alpha=1, linewidth=0.3)
            if axis_snap and np.all(np.isfinite(traces_s[0])):
                ax_s.set_ylim([np.min(traces_s[0])-np.ptp(traces_s[0])*0.05, np.max(traces_s[0])+np.ptp(traces_s[0])*0.05])
                ax_m.set_ylim([np.min(traces_m[0])-np.ptp(traces_m[0])*0.05, np.max(traces_m[0])+np.ptp(traces_m[0])*0.05])
            

            ax_s.grid(False)
            ax_m.grid(False)
            ax_s.set_xlim([0, len(traces_s[0])-1])
            ax_m.set_xlim([0, len(traces_m[0])-1])

            ax_s.set_xticklabels([])
            ax_s.set_yticklabels([])

            ax_m.set_xticklabels([])
            ax_m.set_yticklabels([])
            pbar_inner.update(1)
    fig_synch.tight_layout(pad=0.05)
    fig_mid.tight_layout(pad=0.05)
    if save_location is not None:
        fig_synch.savefig(f'{save_location}/neural_dynamics_synch.pdf', dpi=200)
        fig_synch.savefig(f'{save_location}/neural_dynamics_synch.png', dpi=200)
        fig_mid.savefig(f'{save_location}/neural_dynamics_other.pdf', dpi=200)
        fig_mid.savefig(f'{save_location}/neural_dynamics_other.png', dpi=200)
        plt.close(fig_synch)
        plt.close(fig_mid)
    return fig_synch, fig_mid, which_neurons_mid, mid_colours

def parse_yolo_labels(label_contents, image_width, image_height):
    """
    Parse YOLO format labels and convert to pixel coordinates.
    
    Args:
        label_contents (list): List of label lines from the txt file
        image_width (int): Width of the image in pixels
        image_height (int): Height of the image in pixels
    
    Returns:
        list: List of dictionaries with parsed bounding box info
    """
    bboxes = []
    for line in label_contents:
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                class_id = int(parts[0])
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])
                
                # Convert normalized coordinates to pixel coordinates
                x_center_pixel = x_center_norm * image_width
                y_center_pixel = y_center_norm * image_height
                width_pixel = width_norm * image_width
                height_pixel = height_norm * image_height
                
                # Calculate top-left corner coordinates
                x_min = x_center_pixel - width_pixel / 2
                y_min = y_center_pixel - height_pixel / 2
                x_max = x_center_pixel + width_pixel / 2
                y_max = y_center_pixel + height_pixel / 2
                
                bboxes.append({
                    'class_id': class_id,
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max,
                    'width': width_pixel,
                    'height': height_pixel,
                    'x_center': x_center_pixel,
                    'y_center': y_center_pixel
                })
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse label line '{line}': {e}")
                continue
    return bboxes

def create_image_with_bboxes(image, bboxes, save_path, class_labels=None):
    """
    Create and save an image with bounding boxes overlaid.
    
    Args:
        image (numpy.ndarray): Image array in format (H, W, C) with values 0-1
        bboxes (list): List of bounding box dictionaries from parse_yolo_labels
        save_path (str): Path to save the output PNG
        class_labels (list, optional): List of class names for labels
    """
    try:
        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display the image
        ax.imshow(image)
        ax.axis('off')
        
        # Define colors for different classes (cycling through a palette)
        colors = plt.cm.Set3(np.linspace(0, 1, 12))  # 12 distinct colors
        
        # Draw each bounding box
        for i, bbox in enumerate(bboxes):
            class_id = bbox['class_id']
            color = colors[class_id % len(colors)]
            
            # Create rectangle patch
            rect = Rectangle(
                (bbox['x_min'], bbox['y_min']),
                bbox['width'], 
                bbox['height'],
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add class label if available
            if class_labels and class_id < len(class_labels):
                label_text = class_labels[class_id]
            else:
                label_text = f"Class {class_id}"
            
            # Add text label with background
            ax.text(
                bbox['x_min'], 
                bbox['y_min'] - 5, 
                label_text,
                fontsize=10,
                color='white',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                verticalalignment='top'
            )
        
        # Add title
        title = f"Image with {len(bboxes)} bounding box{'es' if len(bboxes) != 1 else ''}"
        ax.set_title(title, fontsize=14, pad=20)
        
        # Save the figure
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Image with bounding boxes saved to: {save_path}")
        
    except Exception as e:
        print(f"Error creating image with bounding boxes: {e}")

def compute_attention_diagnostics(attention_weights, expected_heads=None):
    """
    Compute attention head similarity and diversity metrics.
    
    Args:
        attention_weights: numpy array of shape (batch, heads, seq_len) 
                          or (heads, seq_len) for single batch
                          or (batch, heads, height, width) for spatial attention
                          or (heads, height, width) for spatial attention single batch
    
    Returns:
        dict with diagnostic metrics
    """
    # Handle different input shapes with better logic
    if len(attention_weights.shape) == 3:
        # Could be (batch, heads, seq_len) or (heads, height, width)
        if expected_heads is not None and attention_weights.shape[0] == expected_heads:
            # Shape is (heads, height, width) - this is what we expect!
            attn = attention_weights.reshape(attention_weights.shape[0], -1)  # Flatten: (heads, height*width)
        else:
            # Shape is (batch, heads, seq_len) - take first batch  
            attn = attention_weights[0]  # Shape: (heads, seq_len)
    else:
        # Handle other shapes (e.g., (heads, seq_len) or (batch, heads, seq_len))
        attn = attention_weights

    n_heads, seq_len = attn.shape
    
    # Ensure we're computing correlations between heads (rows), not positions (columns)
    # np.corrcoef computes correlations between rows by default
    # So attn should be (n_heads, seq_len) where each row is an attention head's pattern
    correlations = np.corrcoef(attn)
    
    # Handle edge case where we only have 1 head (correlation matrix would be scalar)
    if n_heads == 1:
        correlations = np.array([[1.0]])  # Perfect self-correlation
    
    # Ensure correlation matrix is the right size
    assert correlations.shape == (n_heads, n_heads), f"Expected correlation matrix of shape ({n_heads}, {n_heads}), got {correlations.shape}"
    
    # Find redundant head pairs (correlation > 0.8)
    redundant_pairs = []
    redundancy_scores = []
    for i in range(n_heads):
        for j in range(i+1, n_heads):
            corr = correlations[i, j]
            if corr > 0.8:
                redundant_pairs.append((i, j))
                redundancy_scores.append(corr)
    
    # Compute attention entropy for each head (diversity measure)
    entropies = []
    for i in range(n_heads):
        attn_head = attn[i] + 1e-8  # Avoid log(0)
        attn_head = attn_head / attn_head.sum()  # Normalize
        entropy = -np.sum(attn_head * np.log(attn_head))
        entropies.append(entropy)
    
    # Compute attention spread (how concentrated attention is)
    spreads = []
    for i in range(n_heads):
        attn_head = attn[i]
        # Compute standard deviation of attention weights
        spread = np.std(attn_head)
        spreads.append(spread)
    
    # Overall redundancy score (mean pairwise correlation)
    if n_heads > 1:
        mean_correlation = np.mean(correlations[np.triu_indices(n_heads, k=1)])
    else:
        mean_correlation = 0.0  # No pairs to correlate
    
    return {
        'correlations': correlations,
        'redundant_pairs': redundant_pairs,
        'redundancy_scores': redundancy_scores,
        'entropies': entropies,
        'spreads': spreads,
        'mean_correlation': mean_correlation,
        'num_redundant_pairs': len(redundant_pairs)
    }

def make_classification_gif(image, target, predictions, certainties, post_activations, attention_tracking, class_labels, save_location, dataset=None, sample_index=None):
    cmap_viridis = sns.color_palette('viridis', as_cmap=True)
    cmap_spectral = sns.color_palette("Spectral", as_cmap=True)
    figscale = 2
    with tqdm(total=post_activations.shape[0]+1, initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:
        pbar_inner.set_description('Computing UMAP')
    

        low = np.percentile(post_activations, 1, axis=0, keepdims=True)
        high = np.percentile(post_activations, 99, axis=0, keepdims=True)
        post_activations_normed = np.clip((post_activations - low)/(high - low), 0, 1)
        metric = 'cosine'
        reducer = umap.UMAP(n_components=2,
                            n_neighbors=100,
                            min_dist=3,
                            spread=3.0,
                            metric=metric,
                            random_state=None,
                            # low_memory=True,
                            ) if post_activations.shape[-1] > 2048 else umap.UMAP(n_components=2,
                            n_neighbors=20,
                            min_dist=1,
                            spread=1.0,
                            metric=metric,
                            random_state=None,
                            # low_memory=True,
                            )
        positions = reducer.fit_transform(post_activations_normed.T)

        x_umap = positions[:, 0]
        y_umap = positions[:, 1]

        pbar_inner.update(1)
        pbar_inner.set_description('Iterating through to build frames')


        
        frames = []
        route_steps = {}
        route_colours = []

        n_steps = len(post_activations)
        n_heads = attention_tracking.shape[1]
        step_linspace = np.linspace(0, 1, n_steps)
        
        for stepi in np.arange(0, n_steps, 1):
            pbar_inner.set_description('Making frames for gif')
            

            attention_now = attention_tracking[max(0, stepi-5):stepi+1].mean(0)  # Make it smooth for pretty
            # attention_now[:,0,0] = 0  # Corners can be weird looking
            # attention_now[:,0,-1] = 0
            # attention_now[:,-1,0] = 0
            # attention_now[:,-1,-1] = 0
            # attention_now = (attention_tracking[:stepi+1, 0] * decay).sum(0)/(decay.sum(0))
            certainties_now = certainties[1, :stepi+1]
            attention_interp = torch.nn.functional.interpolate(torch.from_numpy(attention_now).unsqueeze(0), image.shape[:2], mode='bilinear')[0]
            attention_interp = (attention_interp.flatten(1) - attention_interp.flatten(1).min(-1, keepdim=True)[0])/(attention_interp.flatten(1).max(-1, keepdim=True)[0] - attention_interp.flatten(1).min(-1, keepdim=True)[0])
            attention_interp = attention_interp.reshape(n_heads, image.shape[0], image.shape[1])
            

            colour = list(cmap_spectral(step_linspace[stepi]))
            route_colours.append(colour)
            for headi in range(min(8, n_heads)):
                com_attn = np.copy(attention_interp[headi])
                com_attn[com_attn < np.percentile(com_attn, 97)] = 0.0
                if headi not in route_steps:
                    A = attention_interp[headi].detach().cpu().numpy()
                    centres, areas = find_island_centers(A, threshold=0.7)
                    route_steps[headi] = [centres[np.argmax(areas)]]
                else:
                    A = attention_interp[headi].detach().cpu().numpy()
                    centres, areas = find_island_centers(A, threshold=0.7)
                    route_steps[headi] = route_steps[headi] + [centres[np.argmax(areas)]]

            mosaic = [['head_0', 'head_0_overlay', 'head_1', 'head_1_overlay'],
                      ['head_2', 'head_2_overlay', 'head_3', 'head_3_overlay'],
                      ['head_4', 'head_4_overlay', 'head_5', 'head_5_overlay'],
                      ['head_6', 'head_6_overlay', 'head_7', 'head_7_overlay'],
                      ['probabilities', 'probabilities','certainty', 'certainty'],
                      ['umap', 'umap', 'umap', 'umap'],
                      ['umap', 'umap', 'umap', 'umap'],
                      ['umap', 'umap', 'umap', 'umap'],
                      ['diagnostics_corr', 'diagnostics_corr', 'diagnostics_text', 'diagnostics_text'],
                      ['diagnostics_corr', 'diagnostics_corr', 'diagnostics_text', 'diagnostics_text'],
                     
                      ]
            
            
            img_aspect = image.shape[0]/image.shape[1]
            # print(img_aspect)
            aspect_ratio = (4*figscale, 10*figscale*img_aspect)  # Increased height for diagnostics
            fig, axes = plt.subplot_mosaic(mosaic, figsize=aspect_ratio)
            for ax in axes.values():
                ax.axis('off')


            axes['certainty'].plot(np.arange(len(certainties_now)), certainties_now, 'k-', linewidth=figscale*1, label='1-(normalised entropy)')
            for ii, (x, y) in enumerate(zip(np.arange(len(certainties_now)), certainties_now)):
                is_correct = predictions[:, ii].argmax(-1)==target
                if is_correct: axes['certainty'].axvspan(ii, ii + 1, facecolor='limegreen', edgecolor=None, lw=0, alpha=0.3)
                else:
                    axes['certainty'].axvspan(ii, ii + 1, facecolor='orchid', edgecolor=None, lw=0, alpha=0.3)
            axes['certainty'].plot(len(certainties_now)-1, certainties_now[-1], 'k.', markersize=figscale*4)
            axes['certainty'].axis('off')
            axes['certainty'].set_ylim([-0.05, 1.05])
            axes['certainty'].set_xlim([0, certainties.shape[-1]+1])

            ps = torch.softmax(torch.from_numpy(predictions[:, stepi]), -1)
            k = 15 if len(class_labels) > 15 else len(class_labels)
            topk = torch.topk (ps, k, dim = 0, largest=True).indices.detach().cpu().numpy()
            top_classes = np.array(class_labels)[topk]
            true_class = target
            colours = [('b' if ci != true_class else 'g') for ci in topk]
            bar_heights = ps[topk].detach().cpu().numpy()


            axes['probabilities'].bar(np.arange(len(bar_heights))[::-1], bar_heights, color=np.array(colours), alpha=1)
            axes['probabilities'].set_ylim([0, 1])
            axes['probabilities'].axis('off')


            for i, (name) in enumerate(top_classes):
                prob = ps[i]
                is_correct = name==class_labels[true_class]
                fg_color = 'darkgreen' if is_correct else 'crimson'
                text_str = f'{name[:40]}'
                axes['probabilities'].text(
                    0.05,
                    0.95 - i * 0.055,  # Adjust vertical position for each line
                    text_str,
                    transform=axes['probabilities'].transAxes,
                    verticalalignment='top',
                    fontsize=8,  # Increased font size
                    color=fg_color,
                    alpha=0.5,
                    path_effects=[
                        patheffects.Stroke(linewidth=3, foreground='aliceblue'),
                        patheffects.Normal()
                    ])
            


            attention_now = attention_tracking[max(0, stepi-5):stepi+1].mean(0)  # Make it smooth for pretty
            # attention_now = (attention_tracking[:stepi+1, 0] * decay).sum(0)/(decay.sum(0))
            certainties_now = certainties[1, :stepi+1]
            attention_interp = torch.nn.functional.interpolate(torch.from_numpy(attention_now).unsqueeze(0), image.shape[:2], mode='nearest')[0]
            attention_interp = (attention_interp.flatten(1) - attention_interp.flatten(1).min(-1, keepdim=True)[0])/(attention_interp.flatten(1).max(-1, keepdim=True)[0] - attention_interp.flatten(1).min(-1, keepdim=True)[0])
            attention_interp = attention_interp.reshape(n_heads, image.shape[0], image.shape[1])

            for hi in range(min(8, n_heads)):
                ax = axes[f'head_{hi}']
                img_to_plot = cmap_viridis(attention_interp[hi].detach().cpu().numpy())
                ax.imshow(img_to_plot)

                ax_overlay = axes[f'head_{hi}_overlay']

                these_route_steps = route_steps[hi]
                y_coords, x_coords = zip(*these_route_steps)
                y_coords = image.shape[-2] - np.array(list(y_coords))-1
                
                ax_overlay.imshow(np.flip(image, axis=0), origin='lower')
                # ax.imshow(np.flip(solution_maze, axis=0), origin='lower')
                arrow_scale = 1.5 if image.shape[0] > 32 else 0.8
                for i in range(len(these_route_steps)-1):
                    dx = x_coords[i+1] - x_coords[i]
                    dy = y_coords[i+1] - y_coords[i]

                    ax_overlay.arrow(x_coords[i], y_coords[i], dx, dy, linewidth=1.6*arrow_scale*1.3, head_width=1.9*arrow_scale*1.3, head_length=1.4*arrow_scale*1.45, fc='white', ec='white', length_includes_head = True, alpha=1)
                    ax_overlay.arrow(x_coords[i], y_coords[i], dx, dy, linewidth=1.6*arrow_scale, head_width=1.9*arrow_scale, head_length=1.4*arrow_scale, fc=route_colours[i], ec=route_colours[i], length_includes_head = True)
                    
                ax_overlay.set_xlim([0,image.shape[1]-1])
                ax_overlay.set_ylim([0,image.shape[0]-1])
                ax_overlay.axis('off')


            z = post_activations_normed[stepi]

            axes['umap'].scatter(x_umap, y_umap, s=30, c=cmap_spectral(z))
            
            # Add attention diagnostics
            # Get current attention weights for all heads (shape: batch, heads, seq_len)
            current_attention = attention_tracking[stepi]  # Shape: (batch, heads, seq_len)
            
            # Compute diagnostics
            diagnostics = compute_attention_diagnostics(current_attention, expected_heads=n_heads)
            
            # Plot correlation matrix heatmap
            corr_matrix = diagnostics['correlations']
            im = axes['diagnostics_corr'].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            axes['diagnostics_corr'].set_title('Head Correlations', fontsize=10, color='white')
            
            # Set head labels
            head_labels = [f'H{i}' for i in range(n_heads)]
            axes['diagnostics_corr'].set_xticks(range(n_heads))
            axes['diagnostics_corr'].set_yticks(range(n_heads))
            axes['diagnostics_corr'].set_xticklabels(head_labels, fontsize=8, color='white')
            axes['diagnostics_corr'].set_yticklabels(head_labels, fontsize=8, color='white')
            
            # Display diagnostic text metrics
            axes['diagnostics_text'].axis('off')
            
            # Prepare diagnostic text with color coding
            diag_text = []
            diag_text.append(f"Step: {stepi+1}/{n_steps}")
            
            # Color-coded mean correlation interpretation
            mean_corr = diagnostics['mean_correlation']
            if mean_corr < 0.3:
                corr_color = 'limegreen'  # Good diversity
                corr_status = "GOOD DIVERSITY"
            elif mean_corr < 0.6:
                corr_color = 'orange'      # Some redundancy
                corr_status = "SOME REDUNDANCY"
            else:
                corr_color = 'red'         # High redundancy
                corr_status = "HIGH REDUNDANCY"
            
            # Color-coded redundant pairs interpretation  
            num_redundant = diagnostics['num_redundant_pairs']
            total_possible_pairs = n_heads * (n_heads - 1) // 2
            if num_redundant == 0:
                pairs_color = 'limegreen'  # Perfect
                pairs_status = "PERFECT"
            elif num_redundant <= 2:
                pairs_color = 'gold'       # Mild concern
                pairs_status = "MILD CONCERN"
            else:
                pairs_color = 'red'        # Problem
                pairs_status = "PROBLEMATIC"
            
            # Create colored text sections
            diag_text.append(f"Mean Correlation: {mean_corr:.3f}")
            diag_text.append(f"Status: {corr_status}")
            diag_text.append("")  # Spacer
            diag_text.append(f"Redundant Pairs: {num_redundant}/{total_possible_pairs}")
            diag_text.append(f"Status: {pairs_status}")
            
            # Add entropy info (head diversity)
            avg_entropy = np.mean(diagnostics['entropies'])
            min_entropy = np.min(diagnostics['entropies'])
            max_entropy = np.max(diagnostics['entropies'])
            diag_text.append("")  # Spacer
            diag_text.append(f"Avg Entropy: {avg_entropy:.3f}")
            diag_text.append(f"Entropy Range: {min_entropy:.3f}-{max_entropy:.3f}")
            
            # Low entropy heads (most focused)
            low_entropy_heads = np.where(np.array(diagnostics['entropies']) < avg_entropy - np.std(diagnostics['entropies']))[0]
            if len(low_entropy_heads) > 0:
                diag_text.append(f"Focused Heads: {[f'H{h}' for h in low_entropy_heads]}")
            
            # High entropy heads (most exploratory) 
            high_entropy_heads = np.where(np.array(diagnostics['entropies']) > avg_entropy + np.std(diagnostics['entropies']))[0]
            if len(high_entropy_heads) > 0:
                diag_text.append(f"Exploratory Heads: {[f'H{h}' for h in high_entropy_heads]}")
            
            # Display main text
            main_text = '\n'.join(diag_text[:2])  # Step and mean correlation
            axes['diagnostics_text'].text(0.05, 0.95, main_text, 
                                        transform=axes['diagnostics_text'].transAxes,
                                        verticalalignment='top', horizontalalignment='left',
                                        fontsize=9, color='white',
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
            
            # Display correlation status with colored background
            axes['diagnostics_text'].text(0.05, 0.80, f"Status: {corr_status}", 
                                        transform=axes['diagnostics_text'].transAxes,
                                        verticalalignment='top', horizontalalignment='left',
                                        fontsize=8, color='black', weight='bold',
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor=corr_color, alpha=0.8))
            
            # Display redundant pairs info
            pairs_text = f"Redundant Pairs: {num_redundant}/{total_possible_pairs}"
            axes['diagnostics_text'].text(0.05, 0.65, pairs_text, 
                                        transform=axes['diagnostics_text'].transAxes,
                                        verticalalignment='top', horizontalalignment='left',
                                        fontsize=9, color='white',
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
            
            # Display pairs status with colored background
            axes['diagnostics_text'].text(0.05, 0.50, f"Status: {pairs_status}", 
                                        transform=axes['diagnostics_text'].transAxes,
                                        verticalalignment='top', horizontalalignment='left',
                                        fontsize=8, color='black', weight='bold',
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor=pairs_color, alpha=0.8))
            
            # Display entropy and head type info
            entropy_text = '\n'.join(diag_text[6:])  # Entropy info and head types
            if entropy_text.strip():  # Only display if there's content
                axes['diagnostics_text'].text(0.05, 0.35, entropy_text, 
                                            transform=axes['diagnostics_text'].transAxes,
                                            verticalalignment='top', horizontalalignment='left',
                                            fontsize=8, color='lightgray',
                                            bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.6))
            
            # Add legend/interpretation guide
            legend_text = "Green: Good  Yellow: Concern  Red: Problem"
            axes['diagnostics_text'].text(0.05, 0.05, legend_text, 
                                        transform=axes['diagnostics_text'].transAxes,
                                        verticalalignment='bottom', horizontalalignment='left',
                                        fontsize=7, color='white', style='italic',
                                        bbox=dict(boxstyle="round,pad=0.2", facecolor='darkgray', alpha=0.7))
        
            fig.tight_layout(pad=0.1)
            


            canvas = fig.canvas
            canvas.draw()
            image_numpy = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
            image_numpy = (image_numpy.reshape(*reversed(canvas.get_width_height()), 4)[:,:,:3])
            frames.append(image_numpy)
            plt.close(fig)
            pbar_inner.update(1)
        pbar_inner.set_description('Saving gif')
        imageio.mimsave(save_location, frames, fps=15, loop=100)
        
        # Save classification data to JSON file
        json_location = save_location.replace('.gif', '.json')
        
        # Prepare data for JSON export
        json_data = {
            'target_class_index': int(target),
            'target_class_name': class_labels[target] if target < len(class_labels) else f"Unknown_{target}",
            'class_labels': class_labels,
            'image_shape': list(image.shape),
            'n_internal_iterations': int(predictions.shape[1]),
            'n_attention_heads': int(attention_tracking.shape[1]),
            'predictions_per_timestep': [],
            'certainties_per_timestep': certainties.tolist(),
            'final_prediction': {
                'class_index': int(predictions[:, -1].argmax()),
                'class_name': class_labels[int(predictions[:, -1].argmax())] if int(predictions[:, -1].argmax()) < len(class_labels) else f"Unknown_{int(predictions[:, -1].argmax())}",
                'confidence': float(torch.softmax(torch.from_numpy(predictions[:, -1]), -1).max()),
                'is_correct': bool(predictions[:, -1].argmax() == target)
            },
            'most_certain_timestep': {
                'timestep': int(certainties[1].argmax()),
                'certainty_value': float(certainties[1].max()),
                'prediction_at_most_certain': {
                    'class_index': int(predictions[:, certainties[1].argmax()].argmax()),
                    'class_name': class_labels[int(predictions[:, certainties[1].argmax()].argmax())] if int(predictions[:, certainties[1].argmax()].argmax()) < len(class_labels) else f"Unknown_{int(predictions[:, certainties[1].argmax()].argmax())}",
                    'confidence': float(torch.softmax(torch.from_numpy(predictions[:, certainties[1].argmax()]), -1).max()),
                    'is_correct': bool(predictions[:, certainties[1].argmax()].argmax() == target)
                }
            },
            'accuracy_over_time': [],
            'top_predictions_final_timestep': []
        }
        
        # Add SARD-specific metadata if dataset and sample_index are provided
        if dataset is not None and sample_index is not None:
            from data.custom_datasets import SARDClassificationDataset
            if isinstance(dataset, SARDClassificationDataset):
                try:
                    # Get the sample info from dataset
                    if sample_index < len(dataset.samples):
                        img_path, target_label = dataset.samples[sample_index]
                        
                        # Get label file path
                        img_filename = os.path.basename(img_path)
                        label_filename = os.path.splitext(img_filename)[0] + '.txt'
                        label_path = os.path.join(dataset.label_dir, label_filename)
                        
                        # Read label file contents
                        label_contents = []
                        if os.path.exists(label_path):
                            try:
                                with open(label_path, 'r') as f:
                                    for line in f:
                                        line = line.strip()
                                        if line:  # Skip empty lines
                                            label_contents.append(line)
                            except Exception as e:
                                print(f"Warning: Could not read label file {label_path}: {e}")
                        
                        # Create image with bounding boxes if we have labels
                        if label_contents:
                            # For SARD dataset, we typically have human detection (class 0)
                            sard_class_labels = ['human']  # SARD typically has just one class: human
                            image_height, image_width = image.shape[:2]
                            
                            # Parse YOLO format bounding boxes
                            bboxes = parse_yolo_labels(label_contents, image_width, image_height)
                            
                            if bboxes:
                                # Create PNG with bounding boxes
                                bbox_png_location = save_location.replace('.gif', '_with_bboxes.png')
                                create_image_with_bboxes(image, bboxes, bbox_png_location, sard_class_labels)
                                
                                # Add bounding box info to JSON metadata
                                json_data['sard_metadata']['bounding_boxes'] = []
                                for bbox in bboxes:
                                    json_data['sard_metadata']['bounding_boxes'].append({
                                        'class_id': bbox['class_id'],
                                        'class_name': sard_class_labels[bbox['class_id']] if bbox['class_id'] < len(sard_class_labels) else f"Class_{bbox['class_id']}",
                                        'x_center_normalized': bbox['x_center'] / image_width,
                                        'y_center_normalized': bbox['y_center'] / image_height,
                                        'width_normalized': bbox['width'] / image_width,
                                        'height_normalized': bbox['height'] / image_height,
                                        'x_min_pixel': bbox['x_min'],
                                        'y_min_pixel': bbox['y_min'],
                                        'x_max_pixel': bbox['x_max'],
                                        'y_max_pixel': bbox['y_max']
                                    })
                                json_data['sard_metadata']['bbox_image_path'] = bbox_png_location
                            else:
                                print(f"Warning: Label file exists but no valid bounding boxes found: {label_path}")
                        
                        # Add SARD metadata to JSON
                        json_data['sard_metadata'] = json_data.get('sard_metadata', {})
                        json_data['sard_metadata'].update({
                            'image_path': img_path,
                            'label_path': label_path,
                            'label_file_exists': os.path.exists(label_path),
                            'label_contents': label_contents,
                            'dataset_split': dataset.split,
                            'sample_index': int(sample_index),
                            'has_human_label': bool(target_label)
                        })
                        
                except Exception as e:
                    print(f"Warning: Could not extract SARD metadata for sample {sample_index}: {e}")
        
        # Add predictions for each timestep
        for t in range(predictions.shape[1]):
            probs = torch.softmax(torch.from_numpy(predictions[:, t]), -1)
            pred_class = int(predictions[:, t].argmax())
            json_data['predictions_per_timestep'].append({
                'timestep': t,
                'predicted_class_index': pred_class,
                'predicted_class_name': class_labels[pred_class] if pred_class < len(class_labels) else f"Unknown_{pred_class}",
                'confidence': float(probs.max()),
                'is_correct': bool(pred_class == target),
                'all_class_probabilities': probs.tolist()
            })
            
            # Track accuracy over time
            json_data['accuracy_over_time'].append({
                'timestep': t,
                'is_correct': bool(pred_class == target),
                'certainty': float(certainties[1, t])
            })
        
        # Add top predictions for final timestep
        final_probs = torch.softmax(torch.from_numpy(predictions[:, -1]), -1)
        top_k = min(5, len(class_labels))  # Top 5 or all classes if fewer
        topk_indices = torch.topk(final_probs, top_k, dim=0, largest=True).indices
        for i, idx in enumerate(topk_indices):
            idx = int(idx)
            json_data['top_predictions_final_timestep'].append({
                'rank': i + 1,
                'class_index': idx,
                'class_name': class_labels[idx] if idx < len(class_labels) else f"Unknown_{idx}",
                'probability': float(final_probs[idx]),
                'is_target': bool(idx == target)
            })
        
        # Save JSON file
        try:
            with open(json_location, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Classification data saved to: {json_location}")
        except Exception as e:
            print(f"Warning: Could not save JSON data to {json_location}: {e}")