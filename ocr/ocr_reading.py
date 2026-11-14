import numpy as np

UNIT_LIST = ["bar", "mbar", "millibars", "MPa", "kPa", "psi", "C", "°C", "F", "°F", "%"]


class OCRReading:
    def __init__(self, polygon, reading, confidence):
        self.polygon = np.array(polygon, dtype=np.float32)
        self.reading = str(reading).strip()
        self.confidence = float(confidence)

        if self.is_number():
            self.number = float(self._clean_number_str(self.reading))
        else:
            self.number = None

        self.center = self._get_centroid()

        self.theta = None

    @staticmethod
    def _clean_number_str(s):
        cleaned = str(s).strip().replace(',', '.').replace('—', '-').replace('–', '-').replace('~', '-')
        # Handle detached minus signs like "- 10" or "-  5"
        if cleaned.startswith('- ') or cleaned.startswith('-  '):
            cleaned = '-' + cleaned.lstrip('- ')
        return cleaned

    def _get_centroid(self):
        x_mean = float(np.mean(self.polygon[:, 0]))
        y_mean = float(np.mean(self.polygon[:, 1]))

        return (x_mean, y_mean)

    def is_number(self):
        cleaned = self._clean_number_str(self.reading)
        try:
            float(cleaned)
            return True
        except ValueError:
            return False

    def is_unit(self):
        cleaned = self.reading.lower().strip()
        return cleaned in [unit.lower() for unit in UNIT_LIST]

    def set_polygon(self, polygon):
        self.polygon = np.array(polygon, dtype=np.float32)
        self.center = self._get_centroid()

    def set_theta(self, theta):
        self.theta = theta

    def get_bounding_box(self):
        x_min = float(np.min(self.polygon[:, 0]))
        y_min = float(np.min(self.polygon[:, 1]))
        x_max = float(np.max(self.polygon[:, 0]))
        y_max = float(np.max(self.polygon[:, 1]))

        return (x_min, y_min, x_max, y_max)
