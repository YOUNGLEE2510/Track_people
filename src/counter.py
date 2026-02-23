import math

class PeopleCounter:
    """
    line_type: "horizontal" | "vertical"
    - horizontal: line ngang/nghiêng, trên→dưới = IN, dưới→trên = OUT
    - vertical:   line dọc giữa, trái→phải = IN, phải→trái = OUT
    """
    def __init__(self, line_y, line_angle=0, line_x1=0, line_x2=640, frame_width=640, frame_height=480,
                 line_type="horizontal", line_x=None):
        self.line_type = line_type
        self.line_y = line_y
        self.line_angle = line_angle
        self.line_x1 = line_x1
        self.line_x2 = line_x2
        self.frame_width = frame_width
        self.frame_height = frame_height
        # Vertical: đường dọc tại x = line_x (mặc định giữa khung)
        self.line_x = int(line_x) if line_x is not None else (frame_width // 2)

        if line_type == "vertical":
            # Đường dọc: x - line_x = 0
            self.a = 1
            self.b = 0
            self.c = -self.line_x
        elif line_angle == 0:
            # Horizontal line: y = line_y
            self.a = 0
            self.b = 1
            self.c = -line_y
        else:
            # Angled line: calculate from two points
            angle_rad = math.radians(line_angle)
            center_x = (line_x1 + line_x2) / 2
            center_y = line_y
            dx = math.cos(angle_rad)
            dy = math.sin(angle_rad)
            if abs(dx) > 1e-6:
                tan_angle = dy / dx
                self.a = -tan_angle
                self.b = 1
                self.c = tan_angle * center_x - center_y
            else:
                self.a = 1
                self.b = 0
                self.c = -center_x

        # 3 cấu trúc dữ liệu chính để tránh double count
        self.track_history = {}      # Lưu vị trí trước đó của từng ID: {track_id: (x, y)}
        self.counted_ids = set()     # Lưu ID đã được đếm (cả IN và OUT): {track_id}
        self.direction_state = {}    # Lưu trạng thái phía của ID: {track_id: 'above'|'below'|'left'|'right'}
        
        # Lưu trữ loại đếm cho mỗi ID để debug
        self.count_type = {}  # {track_id: 'in'|'out'}
        
        # Debounce mechanism để tránh đếm sai do nhiễu hoặc dao động nhỏ
        # Đếm số lần liên tiếp có thay đổi trạng thái trước khi confirm crossing
        self.stable_counter = {}  # {track_id: count} - số lần liên tiếp có thay đổi trạng thái
        self.debounce_threshold = 3  # Số lần thay đổi liên tiếp cần thiết để confirm crossing
        
        # Khoảng cách tối thiểu để ngăn đếm khi quay đầu
        # Chỉ đếm khi người đã vượt qua line một khoảng cách đủ xa
        self.MIN_DISTANCE = 40  # pixels

    def _get_side_of_line(self, x, y):
        """Determine which side of the line a point is on"""
        if self.line_type == "vertical":
            return 'left' if x < self.line_x else 'right'
        distance = self.a * x + self.b * y + self.c
        if self.line_angle == 0:
            return 'above' if y < self.line_y else 'below'
        return 'above' if distance < 0 else 'below'
    
    def _get_distance_to_line(self, x, y):
        """Tính khoảng cách từ điểm đến line"""
        if self.line_type == "vertical":
            # Khoảng cách theo trục X cho vertical line
            return abs(x - self.line_x)
        else:
            # Khoảng cách theo trục Y cho horizontal line
            if self.line_angle == 0:
                return abs(y - self.line_y)
            else:
                # Khoảng cách đến đường thẳng nghiêng
                # Sử dụng công thức khoảng cách từ điểm đến đường thẳng: |ax + by + c| / sqrt(a^2 + b^2)
                numerator = abs(self.a * x + self.b * y + self.c)
                denominator = math.sqrt(self.a**2 + self.b**2)
                return numerator / denominator if denominator > 0 else 0

    def update(self, track_id, cx, cy, shared_counter):
        """
        Update counter with person's center position.
        Horizontal: above→below = IN, below→above = OUT.
        Vertical:   left→right = IN, right→left = OUT.
        
        Logic robust chống double count (theo mẫu):
        - Sử dụng 3 cấu trúc dữ liệu: track_history, counted_ids, direction_state
        - Chỉ đếm khi có crossing (thay đổi trạng thái)
        - Reset counted_ids khi crossing ngược lại để cho phép đếm lại
        """
        # Xác định trạng thái hiện tại của track
        current_state = self._get_side_of_line(cx, cy)
        
        # Nếu track_id mới xuất hiện, khởi tạo trạng thái và return
        if track_id not in self.direction_state:
            self.track_history[track_id] = (cx, cy)
            self.direction_state[track_id] = current_state
            return

        # Lấy trạng thái trước đó
        previous_state = self.direction_state[track_id]
        prev_x, prev_y = self.track_history.get(track_id, (cx, cy))

        # Kiểm tra khoảng cách tối thiểu để ngăn đếm khi quay đầu
        distance_to_line = self._get_distance_to_line(cx, cy)

        # Debounce mechanism: chỉ đếm khi có đủ số lần thay đổi liên tiếp
        # Điều này giúp tránh đếm sai do nhiễu hoặc dao động nhỏ ở đường line
        if previous_state != current_state:
            # Có thay đổi trạng thái - tăng counter
            if track_id not in self.stable_counter:
                self.stable_counter[track_id] = 1
            else:
                self.stable_counter[track_id] += 1

            # Chỉ đếm khi đã có đủ số lần thay đổi liên tiếp VÀ khoảng cách đủ xa
            if self.stable_counter[track_id] >= self.debounce_threshold and distance_to_line >= self.MIN_DISTANCE:
                # Confirm crossing - reset counter và tiếp tục xử lý đếm
                self.stable_counter[track_id] = 0
                
                # Chỉ đếm khi có sự thay đổi trạng thái (crossing line) VÀ khoảng cách đủ xa
                # Logic robust: chỉ đếm khi track_id chưa trong counted_ids
                # Reset counted_ids khi crossing ngược lại để cho phép đếm lại
                if self.line_type == "vertical":
                    # Trái→phải = IN, phải→trái = OUT
                    if previous_state == 'left' and current_state == 'right':
                        # Crossing IN: trái → phải
                        # Reset nếu đã đếm OUT trước đó (cho phép đếm lại)
                        if track_id in self.counted_ids and self.count_type.get(track_id) == 'out':
                            self.counted_ids.discard(track_id)
                        # Chỉ đếm nếu chưa được đếm IN
                        if track_id not in self.counted_ids:
                            shared_counter.add_in()
                            self.counted_ids.add(track_id)
                            self.count_type[track_id] = 'in'
                            print(f"[COUNTER] Person ID {track_id} crossed IN (left→right): ({prev_x}, {prev_y}) -> ({cx}, {cy}), distance={distance_to_line:.1f}")
                    elif previous_state == 'right' and current_state == 'left':
                        # Crossing OUT: phải → trái
                        # Reset nếu đã đếm IN trước đó (cho phép đếm lại)
                        if track_id in self.counted_ids and self.count_type.get(track_id) == 'in':
                            self.counted_ids.discard(track_id)
                        # Chỉ đếm nếu chưa được đếm OUT
                        if track_id not in self.counted_ids:
                            shared_counter.add_out()
                            self.counted_ids.add(track_id)
                            self.count_type[track_id] = 'out'
                            print(f"[COUNTER] Person ID {track_id} crossed OUT (right→left): ({prev_x}, {prev_y}) -> ({cx}, {cy}), distance={distance_to_line:.1f}")
                else:
                    # Horizontal: above→below = IN, below→above = OUT
                    if previous_state == "above" and current_state == "below":
                        # Crossing IN: trên → dưới
                        # Reset nếu đã đếm OUT trước đó (cho phép đếm lại)
                        if track_id in self.counted_ids and self.count_type.get(track_id) == 'out':
                            self.counted_ids.discard(track_id)
                        # Chỉ đếm nếu chưa được đếm IN
                        if track_id not in self.counted_ids:
                            shared_counter.add_in()
                            self.counted_ids.add(track_id)
                            self.count_type[track_id] = 'in'
                            print(f"[COUNTER] Person ID {track_id} crossed IN: ({prev_x}, {prev_y}) -> ({cx}, {cy}), distance={distance_to_line:.1f}")
                    elif previous_state == "below" and current_state == "above":
                        # Crossing OUT: dưới → trên
                        # Reset nếu đã đếm IN trước đó (cho phép đếm lại)
                        if track_id in self.counted_ids and self.count_type.get(track_id) == 'in':
                            self.counted_ids.discard(track_id)
                        # Chỉ đếm nếu chưa được đếm OUT
                        if track_id not in self.counted_ids:
                            shared_counter.add_out()
                            self.counted_ids.add(track_id)
                            self.count_type[track_id] = 'out'
                            print(f"[COUNTER] Person ID {track_id} crossed OUT: ({prev_x}, {prev_y}) -> ({cx}, {cy}), distance={distance_to_line:.1f}")
            else:
                # Chưa đủ số lần thay đổi liên tiếp hoặc khoảng cách chưa đủ, chỉ cập nhật trạng thái
                self.track_history[track_id] = (cx, cy)
                self.direction_state[track_id] = current_state
                return
        else:
            # Không có thay đổi - reset counter
            self.stable_counter[track_id] = 0

        # Cập nhật lịch sử và trạng thái sau mỗi frame
        self.track_history[track_id] = (cx, cy)
        self.direction_state[track_id] = current_state

    def cleanup_track(self, track_id):
        """
        Xóa track khỏi memory khi track biến mất.
        Điều này cho phép track được đếm lại nếu xuất hiện lại sau đó.
        """
        if track_id in self.track_history:
            del self.track_history[track_id]
        if track_id in self.direction_state:
            del self.direction_state[track_id]
        if track_id in self.counted_ids:
            self.counted_ids.remove(track_id)
        if track_id in self.count_type:
            del self.count_type[track_id]
        if track_id in self.stable_counter:
            del self.stable_counter[track_id]

    def reset(self):
        """Reset counter state - xóa tất cả cấu trúc dữ liệu bao gồm debounce counter"""
        self.track_history.clear()
        self.counted_ids.clear()
        self.direction_state.clear()
        self.count_type.clear()
        self.stable_counter.clear() 