import base64
import os
import requests
import torch
import torchvision
from typing import Dict, List, Any, Tuple

class OCRHelper:
    @staticmethod
    def extract_elements_from_screenshot(screenshot: bytes) -> Dict[str, Any]:
        url = os.environ.get("OCR_SERVER_ADDRESS")
        if not url:
            raise EnvironmentError("OCR SERVER ADDRESS NOT SET")
            
        encoded_screenshot = base64.b64encode(screenshot).decode("utf-8")
        response = requests.post(url, json={"img_bytes": encoded_screenshot})

        if response.status_code != 200:
            return {
                "error": f"Request failed with status code {response.status_code}",
                "results": [],
            }
        return response.json()

    @staticmethod
    def add_ocr_elements(
        screenshot: bytes,
        tree_elements: List[str],
        preserved_nodes: List[Dict],
        dummy_node_name: str
    ) -> Tuple[List[str], List[Dict]]:
        tree_bboxes = [
            [
                node["position"][0],
                node["position"][1],
                node["position"][0] + node["size"][0],
                node["position"][1] + node["size"][1],
            ]
            for node in preserved_nodes
        ]

        try:
            ocr_results = OCRHelper.extract_elements_from_screenshot(screenshot)
            ocr_bboxes = ocr_results.get("results", [])
        except Exception as e:
            print(f"OCR Error: {e}")
            return tree_elements, preserved_nodes

        if not ocr_bboxes:
            return tree_elements, preserved_nodes

        preserved_nodes_index = len(preserved_nodes)
        for _, (_, content, box) in enumerate(ocr_bboxes):
            box_coords = [
                int(box.get("left", 0)),
                int(box.get("top", 0)),
                int(box.get("right", 0)),
                int(box.get("bottom", 0)),
            ]

            iou = torchvision.ops.box_iou(
                torch.tensor(tree_bboxes),
                torch.tensor([box_coords])
            ).numpy().flatten()

            if max(iou) < 0.1:
                tree_elements.append(
                    f"{preserved_nodes_index}\t{dummy_node_name}\t\t{content}\t\t"
                )
                preserved_nodes.append({
                    "position": (box_coords[0], box_coords[1]),
                    "size": (box_coords[2] - box_coords[0], box_coords[3] - box_coords[1]),
                    "title": "",
                    "text": content,
                    "role": dummy_node_name,
                })
                preserved_nodes_index += 1

        return tree_elements, preserved_nodes
