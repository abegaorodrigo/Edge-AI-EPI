import cv2

from app.core.config import CLASS_COLORS


def draw_annotations(
    image,
    detections,
    alerts,
    total_time_ms=None,
    duration=None,
):
    frame = image.copy()
    _, width, _ = frame.shape

    # EPIs
    for detection in detections:
        class_name = detection.get("classe", "")
        if class_name == "human":
            continue

        confidence = detection.get("confianca", 0.0)
        bbox = detection.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = map(int, bbox)
        color = CLASS_COLORS.get(class_name, (0, 255, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{class_name} ({confidence:.2f})"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame,
            (x1, max(0, y1 - th - 6)),
            (x1 + tw + 4, max(0, y1)),
            color,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x1 + 2, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Pessoas
    compliant_people = 0
    total_people = len(alerts)

    for alert in alerts:
        bbox = alert.get("pessoa_bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = map(int, bbox)
        missing = alert.get("epis_faltando", [])

        if not missing:
            compliant_people += 1
            person_color = (0, 200, 0)
            status_text = "CONFORME [OK]"
        else:
            person_color = (0, 0, 255)
            status_text = f"FALTANDO: {', '.join(missing)}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 3)

        (tw, th), _ = cv2.getTextSize(
            status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        bg_y1 = max(0, y1 - th - 10)
        bg_y2 = max(0, y1)

        cv2.rectangle(
            frame,
            (x1, bg_y1),
            (x1 + tw + 10, bg_y2),
            person_color,
            -1,
        )
        cv2.putText(
            frame,
            status_text,
            (x1 + 5, bg_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # HUD
    cv2.rectangle(frame, (0, 0), (width, 42), (30, 30, 30), -1)

    if total_people == 0:
        compliance_text = "Nenhum trabalhador na cena"
        status_color = (200, 200, 200)
    elif compliant_people == total_people:
        compliance_text = (
            f"STATUS: 100% SEGURO ({compliant_people}/{total_people})"
        )
        status_color = (0, 255, 0)
    else:
        compliance_text = (
            f"ALERTA: {total_people - compliant_people} IRREGULAR(ES) "
            f"({compliant_people}/{total_people})"
        )
        status_color = (0, 0, 255)

    cv2.putText(
        frame,
        compliance_text,
        (15, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
        cv2.LINE_AA,
    )

    if duration and isinstance(duration, dict):
        metrics_text = (
            f"Total: {total_time_ms:.1f}ms "
            f"(Pre: {duration.get('preprocess', 0):.1f} | "
            f"Inf: {duration.get('inference', 0):.1f} | "
            f"Pos: {duration.get('postprocess', 0):.1f})"
        )
        (mw, _), _ = cv2.getTextSize(
            metrics_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        cv2.putText(
            frame,
            metrics_text,
            (width - mw - 15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    elif total_time_ms is not None:
        metrics_text = f"Tempo: {total_time_ms:.1f}ms"
        (mw, _), _ = cv2.getTextSize(
            metrics_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.putText(
            frame,
            metrics_text,
            (width - mw - 15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    return frame
