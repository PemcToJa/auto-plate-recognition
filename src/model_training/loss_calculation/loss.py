import torch
import torch.nn as nn

class get_loss(nn.Module):
    def __init__(self, error_multiplier_for_blank_canvas=0.5, error_multiplier_for_box=5.0):
        super(get_loss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.error_multiplier_for_blank_canvas = error_multiplier_for_blank_canvas
        self.error_multiplier_for_box = error_multiplier_for_box

    def forward(self, predictions, target):
        exists_box = target[..., 0] == 1
        """
        Tutaj wyciągamy komórki w których znajduje się środek tablicy rejestracyjnej
        .shape to [N, 5]
        """
        true_value_predictions = predictions[exists_box]
        true_value_targets = target[exists_box]

        if true_value_predictions.numel() > 0:
            """
            sprawdzamy loss z propability
            """
            true_value_loss = self.mse(true_value_predictions[..., 0], true_value_targets[..., 0])

            predicted_x_y = true_value_predictions[..., 1:3]
            target_x_y = true_value_targets[..., 1:3]
            """
            W tym momencie robimy torch.sqrt z problemu małych i duzych masek i jak funkcja straty by 
            się nimi zajmowała bez tego. Razem z pierwiastkiem w predicted_height_width i 
            target_height_width funkcja straty potem to potęgując do potęgi 2 w mse, traktuje osobno i 
            inaczej przesunięcia bboxa w małych i dużych tablicach rejestracyjnych
            """
            predicted_height_width = torch.sqrt(torch.clamp(true_value_predictions[..., 3:5], min=1e-6))
            target_height_width = torch.sqrt(torch.clamp(true_value_targets[..., 3:5], min=1e-6))

            final_predicted_box = torch.cat([predicted_x_y, predicted_height_width], dim=-1)
            final_target_box = torch.cat([target_x_y, target_height_width], dim=-1)

            box_loss = self.mse(final_predicted_box, final_target_box)
        else:
            true_value_loss = torch.tensor(0.0, device=predictions.device)
            box_loss = torch.tensor(0.0, device=predictions.device)

        false_value_predictions = predictions[~exists_box]
        false_value_targets = target[~exists_box]
        false_value_loss = self.mse(false_value_predictions[..., 0], false_value_targets[..., 0])

        total_loss = (
                self.error_multiplier_for_box * box_loss
                + true_value_loss
                + self.error_multiplier_for_blank_canvas * false_value_loss
        )

        return total_loss, box_loss, true_value_loss, false_value_loss