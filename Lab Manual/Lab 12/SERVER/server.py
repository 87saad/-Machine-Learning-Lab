#!/usr/bin/env python3
"""
YOLO WebSocket Server for Object Detection
Receives images from Unity (Mobile/Quest Pro) and returns detections
"""

import asyncio
import websockets
import json
import cv2
import numpy as np
from ultralytics import YOLO
import time
from io import BytesIO
from PIL import Image

class YOLODetectionServer:
    def __init__(self, model_path='yolov8n.pt', port=8765):
        """
        Initialize YOLO detection server
        
        Args:
            model_path: Path to YOLO model (yolov8n.pt, yolov5s.pt, etc.)
            port: WebSocket server port
        """
        self.port = port
        self.model = YOLO(model_path)
        self.clients = set()
        
        print(f"✓ YOLO model loaded: {model_path}")
        print(f"✓ Server ready on port {port}")
    
    async def handle_client(self, websocket):
        """Handle incoming WebSocket connections"""
        client_ip = websocket.remote_address[0]
        print(f"✓ Client connected: {client_ip}")
        self.clients.add(websocket)
        
        try:
            async for message in websocket:
                # Receive JPEG image bytes
                response = await self.process_frame(message)
                
                # Send back JSON response
                await websocket.send(json.dumps(response))
                
        except websockets.exceptions.ConnectionClosed:
            print(f"✗ Client disconnected: {client_ip}")
        finally:
            self.clients.remove(websocket)
    
    async def process_frame(self, image_bytes):
        """
        Process received frame and return detections
        
        Args:
            image_bytes: JPEG encoded image bytes
            
        Returns:
            dict: Detection results
        """
        try:
            start_time = time.time()
            
            # Decode JPEG to numpy array
            image = Image.open(BytesIO(image_bytes))
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)[0]
            
            # Parse detections
            detections = []
            
            for box in results.boxes:
                # Get box coordinates (normalized)
                x1, y1, x2, y2 = box.xyxyn[0].cpu().numpy()
                
                detection = {
                    'className': results.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'x': float(x1),
                    'y': float(y1),
                    'width': float(x2 - x1),
                    'height': float(y2 - y1)
                }
                
                detections.append(detection)
            
            inference_time = time.time() - start_time
            
            # Log detections
            if detections:
                print(f"Detected {len(detections)} objects in {inference_time:.3f}s")
            
            return {
                'status': 'success',
                'detections': detections,
                'inference_time': inference_time
            }
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'detections': []
            }
    
    async def start(self):
        """Start WebSocket server"""
        print(f"\n{'='*50}")
        print(f"YOLO Detection Server Starting...")
        print(f"{'='*50}")
        print(f"Listening on: ws://0.0.0.0:{self.port}")
        print(f"Connect Unity to: ws://192.168.100.11:{self.port}")
        print(f"{'='*50}\n")
        
        async with websockets.serve(
            self.handle_client, 
            "0.0.0.0", 
            self.port,
            max_size=10*1024*1024  # 10MB max message size
        ):
            await asyncio.Future()  # Run forever


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO WebSocket Detection Server')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model path (yolov8n.pt, yolov5s.pt, etc.)')
    parser.add_argument('--port', type=int, default=8765,
                       help='WebSocket server port')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to run on (0=GPU, cpu=CPU)')
    
    args = parser.parse_args()
    
    # Set device
    import torch
    if args.device != 'cpu' and torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ Using CPU")
        args.device = 'cpu'
    
    # Create and start server
    server = YOLODetectionServer(
        model_path=args.model,
        port=args.port
    )
    
    asyncio.run(server.start())


if __name__ == '__main__':
    main()