from __future__ import annotations


COMPATIBILITY_FIX_ID = (
    "cql_random_actions_device_transfer_v1"
)


def make_device_compatible_cql_trainer(
    official_trainer_class,
):
    """
    Wrap the frozen official CQLTrainer without modifying
    the reference repository.

    The frozen CQL implementation constructs its uniformly
    sampled random actions with torch.FloatTensor(), which
    places them on CPU even when the CQL networks and
    observations are on CUDA.

    This wrapper preserves the sampled action values and
    distribution and only transfers the action tensor to
    the observation device immediately before Q-network
    evaluation.
    """

    class DeviceCompatibleCQLTrainer(
        official_trainer_class
    ):
        def _get_tensor_values(
            self,
            obs,
            actions,
            network=None,
        ):
            if actions.device != obs.device:
                actions = actions.to(
                    device=obs.device
                )

            return super()._get_tensor_values(
                obs,
                actions,
                network=network,
            )

    DeviceCompatibleCQLTrainer.__name__ = (
        "DeviceCompatibleCQLTrainer"
    )

    return DeviceCompatibleCQLTrainer