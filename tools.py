from utils.augmentation import *
import random
from utils.date import *
from database import session, Employee, Attendance, Shift

def add_attendance(id, shift, date, checkin=None, checkout=None):
    new_attendance = Attendance(
        employee_id=id,
        shift=shift,
        date=date
    )

    if checkin:
        new_attendance.checkin = checkin

    if checkout:
        new_attendance.checkout = checkout

    session.add(new_attendance)
    session.commit()
    session.refresh(new_attendance)

    return new_attendance

def checkin(id, time=None, date=None):
    print("\n==============================")
    print("BẮT ĐẦU CHECKIN")
    print("ID nhận từ AI:", id)
    print("Kiểu ID:", type(id))

    # ========================================
    # Ép ID từ SVC về integer
    # ========================================
    try:
        id = int(id)
    except (ValueError, TypeError):
        print("❌ ID không hợp lệ:", id)
        return {
            "success": False,
            "message": "ID nhân viên không hợp lệ"
        }

    print("ID sau khi convert:", id)

    # ========================================
    # Tìm nhân viên
    # ========================================
    employee = (
        session.query(Employee)
        .filter_by(id=id)
        .first()
    )

    print("Employee tìm được:", employee)

    if not employee:
        print("❌ Không tìm thấy nhân viên ID:", id)

        return {
            "success": False,
            "employee_id": id,
            "message": "Không tìm thấy nhân viên"
        }

    print("✅ Nhân viên:", employee.name)

    skip = (
        employee.position.upper() == "NVHC"
        if employee.position
        else False
    )

    current_time = get_time() if time is None else time
    current_date = get_date() if date is None else date

    print("Ngày:", current_date)
    print("Giờ:", current_time)
    print("Position:", employee.position)
    print("Skip:", skip)

    # ========================================
    # Lấy danh sách ca
    # ========================================
    shifts = session.query(Shift).all()

    print("Số ca:", len(shifts))

    if len(shifts) == 0:
        print("❌ Database chưa có ca làm việc")

        return {
            "success": False,
            "employee_id": id,
            "employee_name": employee.name,
            "message": "Chưa thiết lập ca làm việc"
        }

    for shift in shifts:
        print(
            "SHIFT:",
            shift.name,
            shift.checkin,
            "-",
            shift.checkout
        )

    # ========================================
    # Xác định ca hiện tại
    # ========================================
    current_shift = None
    current_index = 0

    for index, shift in enumerate(shifts):
        if shift.checkin <= current_time <= shift.checkout:
            current_shift = shift
            current_index = index
            break

    # Nếu đang ngoài giờ ca
    if current_shift is None:

        index = 0

        for shift in shifts:
            if current_time > shift.checkout:
                index += 1

        if index >= len(shifts):
            index = len(shifts) - 1

        current_shift = shifts[index]
        current_index = index

    print(
        "Ca được chọn:",
        current_shift.name,
        current_shift.checkin,
        current_shift.checkout
    )

    # ========================================
    # Lịch sử hôm nay
    # ========================================
    recs = (
        session.query(Attendance)
        .filter(
            Attendance.employee_id == id,
            Attendance.date == current_date
        )
        .all()
    )

    print("Số bản ghi hôm nay:", len(recs))

    for rec in recs:
        print(
            "ATTENDANCE:",
            rec.shift,
            rec.checkin,
            rec.checkout
        )

    current_his = None
    previous_his = None

    for rec in recs:

        if rec.shift == current_shift.name:
            current_his = rec

        else:
            previous_his = rec

    message = "checkout"

    final_in = None
    final_out = None

    # ========================================
    # NVHC
    # ========================================
    if skip:

        print("Nhân viên NVHC")

        if not recs:

            attendance = add_attendance(
                id,
                current_shift.name,
                current_date,
                checkin=current_time
            )

            recs.append(attendance)

            message = "checkin"

        else:

            attendance = recs[-1]

            if attendance.checkout:

                print("⚠️ Nhân viên đã checkout")

                return {
                    "success": False,
                    "employee_id": id,
                    "employee_name": employee.name,
                    "message": "Nhân viên đã chấm công đầy đủ"
                }

            attendance.checkout = current_time
            message = "checkout"

        final_in = recs[0].checkin
        final_out = recs[-1].checkout

    # ========================================
    # Nhân viên thường
    # ========================================
    else:

        # Chưa có record của ca hiện tại
        if current_his is None:

            # Chưa có lịch sử nào
            if previous_his is None:

                attendance = add_attendance(
                    id,
                    current_shift.name,
                    current_date,
                    checkin=current_time
                )

                recs.append(attendance)

                message = "checkin"

                print("✅ Tạo CHECK-IN mới")

            # Có record ca trước
            else:

                if previous_his.checkout is None:

                    previous_his.checkout = current_time
                    attendance = previous_his
                    message = "checkout"

                    print("✅ CHECK-OUT ca trước")

                else:

                    attendance = add_attendance(
                        id,
                        current_shift.name,
                        current_date,
                        checkin=current_time
                    )

                    recs.append(attendance)

                    message = "checkin"

                    print("✅ CHECK-IN ca mới")

        # Đã có record ca hiện tại
        else:

            attendance = current_his

            if attendance.checkout is None:

                attendance.checkout = current_time
                message = "checkout"

                print("✅ CHECK-OUT")

            else:

                print("⚠️ Ca này đã chấm công đầy đủ")

                return {
                    "success": False,
                    "employee_id": id,
                    "employee_name": employee.name,
                    "message": "Ca này đã chấm công đầy đủ"
                }

    # ========================================
    # Commit
    # ========================================
    session.commit()

    session.refresh(attendance)

    # ========================================
    # Response
    # ========================================
    info = {
        "success": True,
        "employee_id": id,
        "employee_name": employee.name,
        "attendance_date": date_to_string(attendance.date),

        "attendance_checkin":
            time_to_string(attendance.checkin)
            if attendance.checkin
            else None,

        "attendance_checkout":
            time_to_string(attendance.checkout)
            if attendance.checkout
            else None,

        "message": message
    }

    if final_in:
        info["attendance_checkin"] = time_to_string(final_in)

    if final_out:
        info["attendance_checkout"] = time_to_string(final_out)

    # ========================================
    # Trạng thái đi muộn
    # ========================================
    if message == "checkin" and attendance.checkin:

        late = round(
            time_to_minutes(attendance.checkin)
            -
            time_to_minutes(current_shift.checkin)
        )

        if late > 0:
            info["checkin_status"] = f"Muộn {late} phút"
            info["checkin_flag"] = False
        else:
            info["checkin_status"] = "Đúng giờ"
            info["checkin_flag"] = True

    else:

        info["checkin_status"] = "Checkout thành công"
        info["checkin_flag"] = True

    print("✅ CHECKIN RESULT:")
    print(info)
    print("==============================\n")

    return info

def augmentate(image):
    augmentations = [
        lambda img: adjust_brightness(
            img,
            random.choice([20, -20])
        ),
        lambda img: adjust_contrast(
            img,
            random.choice([1.5, 0.7])
        ),
        lambda img: rotate_image(
            img,
            random.choice([5, -5])
        ),
        lambda img: random.choice(
            [add_gaussian_noise, blur_image]
        )(img),
    ]

    augmented_images = [
        augment(image)
        for augment in augmentations
    ]

    return augmented_images