import cv2
import gc
import numpy as np
import os
import os.path as osp
import torch
import argparse
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm
from ultralytics import YOLO

def init_config():
    color = [(255, 0, 0)]
    exp_name = "samurai"
    model_name = "base_plus"
    
    checkpoint = f"sam2/checkpoints/sam2.1_hiera_{model_name}.pt"
    model_cfg = "configs/samurai/sam2.1_hiera_b+.yaml" if model_name == "base_plus" else f"configs/samurai/sam2.1_hiera_{model_name[0]}.yaml"
    
    pred_folder = f"results/{exp_name}/{exp_name}_{model_name}"
    vis_folder = f"visualization/{exp_name}/{model_name}"
    os.makedirs(vis_folder, exist_ok=True)
    os.makedirs(pred_folder, exist_ok=True)
    
    return color, model_cfg, checkpoint, pred_folder, vis_folder

def init_video(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video")
        exit()
        
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    return video, height, width, fps

def detect_people(frame, model):
    results = model(frame)
    prompts = {}
    person_count = 0
    
    for result in results:
        for bbox in result.boxes:
            if int(bbox.cls[0]) == 0:  # Only detect persons
                x1, y1, x2, y2 = bbox.xyxy[0]
                prompts[person_count] = ((int(x1), int(y1), int(x2), int(y2)), 0)
                person_count += 1
                
    return prompts

def select_frame_with_people(video, model):
    print("Press SPACE to select a frame, ESC to exit")
    frame_count = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            print("Reached end of video")
            exit()
            
        frame_count += 1
        cv2.imshow("Select frame (SPACE to choose, ESC to exit)", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            exit()
        elif key == 32:  # SPACE
            prompts = detect_people(frame, model)
            if prompts:
                return frame, prompts, frame_count
            print("No people detected in this frame. Please try another frame.")

def display_and_select_person(frame, prompts):
    img_with_boxes = frame.copy()
    for i, (bbox_coords, cls) in prompts.items():
        x1, y1, x2, y2 = bbox_coords
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_with_boxes, f"Person {i}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Select a person to track", img_with_boxes)
    cv2.waitKey(1)

    while True:
        try:
            choice = int(input(f"Enter person number to track (0-{len(prompts)-1}): "))
            if choice in prompts:
                return prompts[choice]
            print(f"Please enter a valid number between 0 and {len(prompts)-1}")
        except ValueError:
            print("Please enter a valid number")

def extract_frames(video, frame_folder, start_frame):
    frame_idx = 0
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        cv2.imwrite(f'{frame_folder}/{frame_idx:08d}.jpg', frame)
        frame_idx += 1

def process_frame(masks, object_ids, height, width, color):
    mask_to_vis = {}
    bbox_to_vis = {}
    
    for obj_id, mask in zip(object_ids, masks):
        mask = mask[0].cpu().numpy() > 0.0
        non_zero_indices = np.argwhere(mask)
        
        if len(non_zero_indices) == 0:
            bbox = [0, 0, 0, 0]
        else:
            y_min, x_min = non_zero_indices.min(axis=0).tolist()
            y_max, x_max = non_zero_indices.max(axis=0).tolist()
            bbox = [x_min, y_min, x_max-x_min, y_max-y_min]
            
        bbox_to_vis[obj_id] = bbox
        mask_to_vis[obj_id] = mask
        
    return mask_to_vis, bbox_to_vis

def visualize_frame(img, mask_to_vis, bbox_to_vis, height, width, color, yolo_model):
    # Create YOLO detection visualization
    yolo_vis = img.copy()
    results = yolo_model(yolo_vis)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if int(box.cls[0]) == 0:  # Person class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cv2.rectangle(yolo_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Create SAM2 visualization
    sam_vis = img.copy()
    for obj_id in mask_to_vis:
        mask_img = np.zeros((height, width, 3), np.uint8)
        mask_img[mask_to_vis[obj_id]] = color[(obj_id+1)%len(color)]
        sam_vis = cv2.addWeighted(sam_vis, 1, mask_img, 0.75, 0)
    
    for obj_id in bbox_to_vis:
        bbox = bbox_to_vis[obj_id]
        cv2.rectangle(sam_vis, (bbox[0], bbox[1]),
                     (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                     color[obj_id%len(color)], 2)
    
    # Combine visualizations side by side
    combined_vis = np.hstack((yolo_vis, sam_vis))
    cv2.putText(combined_vis, "YOLO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined_vis, "SAM2", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow('Tracking Comparison', combined_vis)
    return cv2.waitKey(1) & 0xFF == ord('q'), combined_vis

def track_with_yolo(video, model, frame_folder, bbox, height, width, color):
    predictions = []
    frame_idx = 0
    state = None
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for frame_idx in tqdm(range(len(os.listdir(frame_folder)))):
            img = cv2.imread(f'{frame_folder}/{frame_idx:08d}.jpg')
            if img is None:
                break
                
            if state is None:
                state = model.init_state(frame_folder, offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)
                frame_idx, object_ids, masks = model.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)
            else:
                frame_idx, object_ids, masks = model.add_new_points_or_box(state, box=None, frame_idx=frame_idx, obj_id=0)
            
            mask_to_vis, bbox_to_vis = process_frame(masks, object_ids, height, width, color)
            should_quit, img_with_vis = visualize_frame(img, mask_to_vis, bbox_to_vis, height, width, color)
            if should_quit:
                break
                
            predictions.append(bbox_to_vis)
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Track people in videos using SAM2')
    parser.add_argument('--video', required=True, help='Path to video file to process')
    args = parser.parse_args()

    color, model_cfg, checkpoint, pred_folder, vis_folder = init_config()
    
    video, height, width, fps = init_video(args.video)
    frame_folder = "temp"
    os.makedirs(frame_folder, exist_ok=True)
    
    model = YOLO('yolo11x.pt')
    frame_with_people, prompts, frame_count = select_frame_with_people(video, model)
    bbox, cls = display_and_select_person(frame_with_people, prompts)
    cv2.destroyAllWindows()
    
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda:0")
    extract_frames(video, frame_folder, frame_count)
    video.release()
    cv2.destroyAllWindows()
    
    predictions = []
    video_basename = osp.basename(args.video).split('.')[0]
    
    # Initialize video writer with mp4v codec for double width output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = osp.join(vis_folder, f'{video_basename}.mp4')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width*2, height))  # Double width for side by side
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frame_folder, offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)
        frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)
        
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            assert len(masks) == 1 and len(object_ids) == 1, "Only one object is supported right now"
            
            mask_to_vis, bbox_to_vis = process_frame(masks, object_ids, height, width, color)
            
            img = cv2.imread(f'{frame_folder}/{frame_idx:08d}.jpg')
            if img is None:
                break
                
            should_quit, img_with_vis = visualize_frame(img, mask_to_vis, bbox_to_vis, height, width, color, model)
            if should_quit:
                break
                
            out.write(img_with_vis)
            predictions.append(bbox_to_vis)
    
    # Make sure to properly close video writer
    out.release()
    
    # Verify the output video was created successfully
    if not osp.exists(out_path):
        print(f"Error: Failed to save video to {out_path}")
    else:
        print(f"Successfully saved video to {out_path}")
    
    # Save bounding box predictions
    pred_path = osp.join(pred_folder, f'{video_basename}.txt')
    with open(pred_path, 'w') as f:
        for pred in predictions:
            x, y, w, h = pred[0]
            f.write(f"{x},{y},{w},{h}\n")
    
    print(f"Saved predictions to {pred_path}")
    
    cv2.destroyAllWindows()
    del predictor
    del state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
