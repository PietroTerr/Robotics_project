import logging
import sys
import time
from datetime import datetime


class SimulationLogger:
    """Handles structured logging for a simulation"""

    def __init__(self, log_interval=1000, log_level=logging.INFO):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        self.log_interval = log_interval
        self.start_time = None
        self.step_count = 0

        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            # Clean, minimal prefix
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def _format_value(self, val):
        """Helper to safely format floats, tuples of floats, or other types."""
        if isinstance(val, float):
            return f"{val:.4f}"
        elif isinstance(val, tuple):
            # Format each item inside the tuple if it's a float
            formatted_items = [f"{x:.4f}" if isinstance(x, float) else str(x) for x in val]
            return f"({', '.join(formatted_items)})"
        else:
            return str(val)

    def start(self, total_steps=None, **params):
        """Logs the start of the simulation"""
        self.start_time = time.time()

        lines = [
            "",
            "=" * 50,
            "🚀 SIMULATION STARTED",
            f"Timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}"
        ]

        if total_steps:
            lines.append(f"Total steps: {total_steps}")

        for key, value in params.items():
            # Clean up the key name for display (e.g., start_pos -> Start Pos)
            clean_key = key.replace('_', ' ').title()
            lines.append(f"  • {clean_key}: {self._format_value(value)}")

        lines.append("=" * 50)
        self.logger.info("\n".join(lines))

    def log_step(self, step, **metrics):
        """Logs information about the current step"""
        self.step_count = step

        if step % self.log_interval == 0:
            elapsed = time.time() - self.start_time

            # Dashboard-style block for the current step
            lines = [
                "",
                "-" * 50,
                f"🔄 STEP {step:05d} | Elapsed Time: {elapsed:.2f}s"
            ]

            for key, value in metrics.items():
                clean_key = key.replace('_', ' ').title()
                # Align the keys to a fixed width (e.g., 18 characters) for a clean table look
                lines.append(f"  • {clean_key:<18}: {self._format_value(value)}")

            lines.append("-" * 50)
            self.logger.info("\n".join(lines))

    def log_event(self, event_type, **details):
        """Logs specific events"""
        detail_str = " | ".join(f"{k}: {self._format_value(v)}" for k, v in details.items())
        if event_type == "warning":
            self.logger.warning(f"EVENT: {detail_str}")
        elif event_type == "error":
            self.logger.error(f"ERROR: {detail_str}")
        else:
            self.logger.info(f"EVENT [{event_type}]: {detail_str}")

    def end(self, reason="Completed", **final_metrics):
        """Logs the end of the simulation"""
        if self.start_time is None:
            self.logger.warning("Simulation ended before it started!")
            return

        total_time = time.time() - self.start_time

        lines = [
            "",
            "=" * 50,
            f"🛑 SIMULATION FINISHED: {reason}",
            f"  • Total steps: {self.step_count}",
            f"  • Total time:  {total_time:.2f}s"
        ]

        if self.step_count > 0:
            lines.append(f"  • Avg per step: {total_time / self.step_count * 1000:.2f}ms")
        lines.append("-" * 20)
        for key, value in final_metrics.items():
            clean_key = key.replace('_', ' ').title()
            lines.append(f"  • {clean_key}: {self._format_value(value)}")

        lines.append("=" * 50)
        self.logger.info("\n".join(lines))