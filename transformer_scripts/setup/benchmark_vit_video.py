import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import cv2
import numpy as np
import time
import os
import argparse
import tensorrt as trt
from tqdm import tqdm

# --- Configuration & Model Mapping ---
MODEL_MAP = {
    "base": "google/vit-base-patch16-224",
    "large": "google/vit-large-patch16-224",
    "huge": "google/vit-huge-patch14-224-in21k"
}

# --- Helper Functions ---

def letterbox_resize(frame: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Resizes an image to a square, preserving aspect ratio by padding.
    This is the correct way to prepare a non-square image for a ViT.
    """
    original_h, original_w = frame.shape[:2]
    
    # Determine the scale factor and new dimensions
    scale = target_size / max(original_h, original_w)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    
    # Resize the image
    resized_img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a black square canvas and paste the resized image in the center
    canvas = np.full((target_size, target_size, 3), 0, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_img
    
    return canvas

def draw_overlay(frame, label, fps):
    """Draws the classification label and FPS counter on the frame."""
    text = f"Prediction: {label}"
    fps_text = f"FPS: {fps:.2f}"
    
    # Use a black rectangle for text background for better readability
    cv2.rectangle(frame, (0, 0), (500, 70), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

# --- Core Inference and Engine Building Logic (Adapted from previous script) ---

def get_pytorch_prediction(model, image_tensor):
    with torch.no_grad():
        logits = model(image_tensor).logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

def get_tensorrt_prediction(context, engine, input_tensor, output_buffer, model_config):
    # Set the tensor addresses for this inference call
    context.set_tensor_address("input", input_tensor.data_ptr())
    context.set_tensor_address("output", output_buffer.data_ptr())
    
    # Execute inference
    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    
    # Get the prediction from the output buffer
    predicted_class_idx = output_buffer.argmax(-1).item()
    return model_config.id2label[predicted_class_idx]

def build_tensorrt_engine(onnx_path, engine_path, use_fp16, batch_size):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if use_fp16: config.set_flag(trt.BuilderFlag.FP16)
        
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors): print(parser.get_error(error))
            raise ValueError("Failed to parse the ONNX file.")
            
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    min_shape = (1, 3, input_tensor.shape[2], input_tensor.shape[3])
    opt_shape = (batch_size, 3, input_tensor.shape[2], input_tensor.shape[3])
    max_shape = (batch_size, 3, input_tensor.shape[2], input_tensor.shape[3])
    profile.set_shape(input_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)
    
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None: raise RuntimeError("Failed to build the TensorRT engine.")
        
    with open(engine_path, "wb") as f: f.write(serialized_engine)


# --- Main Video Processing Function ---

def main():
    parser = argparse.ArgumentParser(description="Video Classification Benchmark Tool with ViT")
    parser.add_argument('--input_video', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--output_video', type=str, required=True, help="Path to save the output video file.")
    
    model_group = parser.add_mutually_exclusive_group(required=False)
    model_group.add_argument('--base', action='store_true', help="Use ViT-Base model (default)")
    model_group.add_argument('--large', action='store_true', help="Use ViT-Large model")
    model_group.add_argument('--huge', action='store_true', help="Use ViT-Huge model")
    
    parser.add_argument('--tensorrt', action='store_true', help="Use TensorRT for inference.")
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp16', 'fp32'], help="Set model precision (fp16 or fp32).")
    
    args = parser.parse_args()

    # --- Setup ---
    model_size = "large" if args.large else "huge" if args.huge else "base"
    model_name = MODEL_MAP[model_size]
    precision = args.precision
    use_fp16 = (precision == 'fp16')
    device = torch.device("cuda")

    print("--- Configuration ---")
    print(f"Input Video: {args.input_video}")
    print(f"Output Video: {args.output_video}")
    print(f"Model: {model_name} @ {precision}")
    print(f"Inference Mode: {'TensorRT' if args.tensorrt else 'PyTorch'}")
    print("---------------------\n")

    # Load Model and Processor
    print("Loading model and processor...")
    model = ViTForImageClassification.from_pretrained(model_name).to(device).eval()
    processor = ViTImageProcessor.from_pretrained(model_name)
    input_size = processor.size['height']
    if use_fp16: model.half()
    
    # --- TensorRT Setup (if enabled) ---
    tensorrt_context = None
    tensorrt_engine = None
    tensorrt_output_buffer = None
    if args.tensorrt:
        onnx_filename = f"vit-{model_size}-{precision}.onnx"
        engine_filename = f"vit-{model_size}-{precision}.engine"
        
        if not os.path.exists(engine_filename):
            print(f"TensorRT engine not found. Building from ONNX...")
            dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
            if use_fp16: dummy_input = dummy_input.half()
            
            print(f"Exporting to ONNX: {onnx_filename}")
            torch.onnx.export(model, dummy_input, onnx_filename, input_names=['input'], output_names=['output'], opset_version=17, dynamic_axes={'input': {0: 'batch_size'}})
            
            print(f"Building TensorRT engine: {engine_filename}. This may take a while...")
            build_tensorrt_engine(onnx_filename, engine_filename, use_fp16, batch_size=1)
        
        print("Loading TensorRT engine...")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_filename, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            tensorrt_engine = runtime.deserialize_cuda_engine(f.read())
        tensorrt_context = tensorrt_engine.create_execution_context()
        tensorrt_output_buffer = torch.empty((1, 1000), dtype=torch.float16 if use_fp16 else torch.float32, device=device)

    # --- Video Processing ---
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Could not open video file {args.input_video}")
    
    # Video properties for output
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
    writer = cv2.VideoWriter(args.output_video, fourcc, original_fps, (original_w, original_h))

    print(f"\nProcessing video ({total_frames} frames)...")
    start_time = time.perf_counter()
    
    # Initialize variables for live FPS calculation
    frame_count = 0
    display_fps = 0

    for _ in tqdm(range(total_frames), desc="Processing Frames"):
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Pre-process Frame
        # BGR (OpenCV) to RGB (PyTorch)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Letterbox resize to model's input size
        preprocessed_frame = letterbox_resize(rgb_frame, target_size=input_size)
        # Use HuggingFace processor to create the tensor
        image_tensor = processor(images=preprocessed_frame, return_tensors="pt").to(device)
        if use_fp16: image_tensor.pixel_values = image_tensor.pixel_values.half()

        # 2. Run Inference
        if args.tensorrt:
            label = get_tensorrt_prediction(tensorrt_context, tensorrt_engine, image_tensor.pixel_values, tensorrt_output_buffer, model.config)
        else:
            label = get_pytorch_prediction(model, image_tensor.pixel_values)

        # 3. Draw Overlay
        # Calculate live FPS to display on video (averaged over last 30 frames)
        frame_count += 1
        current_time = time.perf_counter()
        if (current_time - start_time) > 1e-6: # Avoid division by zero at start
            if frame_count % 30 == 0: # Update FPS display every 30 frames
                 display_fps = frame_count / (current_time - start_time)

        annotated_frame = draw_overlay(frame, label, display_fps)

        # 4. Write Frame
        writer.write(annotated_frame)
    
    end_time = time.perf_counter()
    
    # Cleanup
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    # --- Final Summary ---
    total_time = end_time - start_time
    processed_fps = total_frames / total_time
    print("\n\n--- PROCESSING COMPLETE ---")
    print(f"Total frames processed: {total_frames}")
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average processing speed: {processed_fps:.2f} FPS")
    print(f"Output video saved to: {args.output_video}")
    print("---------------------------\n")

if __name__ == "__main__":
    main()