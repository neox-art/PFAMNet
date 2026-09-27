import unittest

import torch

from models import PFAMNet


class PFAMNetTests(unittest.TestCase):
    def test_model_name(self):
        self.assertEqual(PFAMNet.__name__, 'PFAMNet')

    def test_outputs_and_checkpoint_loading(self):
        for deep_supervision in (False, True):
            with self.subTest(deep_supervision=deep_supervision):
                model = PFAMNet(
                    backbone_weight_path=None,
                    use_deep_supervision=deep_supervision,
                ).eval()
                restored = PFAMNet(
                    backbone_weight_path=None,
                    use_deep_supervision=deep_supervision,
                ).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                inputs = torch.randn(1, 3, 64, 64)
                with torch.inference_mode():
                    outputs = model(inputs)
                    restored_outputs = restored(inputs)
                channels = [1, 2, 1] + ([1] * 4 if deep_supervision else [])
                self.assertEqual(len(outputs), len(channels))
                for output, restored_output, channel in zip(
                    outputs, restored_outputs, channels
                ):
                    self.assertEqual(tuple(output.shape), (1, channel, 64, 64))
                    self.assertTrue(torch.isfinite(output).all().item())
                    torch.testing.assert_close(output, restored_output)


if __name__ == '__main__':
    unittest.main()
