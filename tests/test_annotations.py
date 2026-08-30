import unittest

from backend.annotations import coco_label_from_yolo_id, parse_yolo_dataset_yaml, parse_yolo_label_text


class AnnotationTests(unittest.TestCase):
    def test_standard_coco80_mapping_is_contiguous(self):
        self.assertEqual(coco_label_from_yolo_id(0), "person")
        self.assertEqual(coco_label_from_yolo_id(11), "stop sign")
        self.assertEqual(coco_label_from_yolo_id(79), "toothbrush")

    def test_dataset_yaml_names_override_defaults(self):
        names = parse_yolo_dataset_yaml("names: [widget, component]\n")
        self.assertEqual(names, ["widget", "component"])
        boxes = parse_yolo_label_text("1 0.5 0.5 0.2 0.4", 100, 100, names)
        self.assertEqual(boxes[0].label, "component")


if __name__ == "__main__":
    unittest.main()
