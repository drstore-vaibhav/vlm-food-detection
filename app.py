"""Streamlit front-end for the LangGraph dietitian agent."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4
import re

import pandas as pd
import streamlit as st

from logic import build_data_url, detect_food_items, enrich_food_items

LOG_FILE = Path("meal_logs.json")
MEAL_TYPES = ["Breakfast", "Lunch", "Snack", "Dinner"]
FIELD_ALIASES = {
    "item": "name",
    "dish": "name",
    "serving_size": "serving size",
    "serving": "serving size",
    "fats": "fats & lipids",
    "fat": "fats & lipids",
    "vitamin_c": "vitamin C",
    "vitaminC": "vitamin C",
    "vitamin_a": "vitamin A",
    "vitaminA": "vitamin A",
    "vitamin_d": "vitamin D",
    "vitaminD": "vitamin D",
    "vitamin_k": "vitamin K",
    "vitamin_k1": "vitamin K",
    "vitaminK": "vitamin K",
    "vitamin_b1": "vitamin B1",
    "vitaminB1": "vitamin B1",
    "thiamin": "vitamin B1",
    "vitamin_b6": "vitamin B6",
    "vitaminB6": "vitamin B6",
    "vitamin_b12": "vitamin B12",
    "vitaminB12": "vitamin B12",
    "vitamin_e": "vitamin E",
    "vitaminE": "vitamin E",
}
NUTRIENT_KEYS = [
    ("protein", "Protein (g)"),
    ("carbs", "Carbs (g)"),
    ("fiber", "Fiber (g)"),
    ("fats & lipids", "Fats & lipids (g)"),
    ("calories", "Calories (kcal)"),
    ("calcium", "Calcium (mg)"),
    ("iron", "Iron (mg)"),
    ("potassium", "Potassium (mg)"),
    ("vitamin C", "Vitamin C (mg)"),
    ("vitamin A", "Vitamin A (IU)"),
    ("vitamin D", "Vitamin D (IU)"),
    ("vitamin K", "Vitamin K (mcg)"),
    ("vitamin B1", "Vitamin B1 (mg)"),
    ("vitamin B6", "Vitamin B6 (mg)"),
    ("vitamin B12", "Vitamin B12 (mcg)"),
    ("vitamin E", "Vitamin E (mg)"),
]


def apply_field_aliases(data: Dict[str, Any]) -> Dict[str, Any]:
    for key in list(data.keys()):
        str_key = str(key)
        alias = FIELD_ALIASES.get(str_key)
        if not alias:
            alias = FIELD_ALIASES.get(str_key.lower())
        if alias and alias not in data:
            data[alias] = data[key]
    return data


def load_logs() -> Dict[str, List[Dict[str, Any]]]:
    if LOG_FILE.exists():
        try:
            return json.loads(LOG_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_logs(logs: Dict[str, List[Dict[str, Any]]]) -> None:
    LOG_FILE.write_text(json.dumps(logs, indent=2))


def ensure_state() -> None:
    st.session_state.setdefault("detected_items", [])
    st.session_state.setdefault("raw_detection", "")
    st.session_state.setdefault("raw_nutrition", "")
    st.session_state.setdefault("editing_rows", set())
    st.session_state.setdefault("meal_type", MEAL_TYPES[0])
    st.session_state.setdefault("nutrition_ready", False)
    st.session_state.setdefault("last_image_preview", None)


def normalize_items(raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in raw_items:
        clean = apply_field_aliases(dict(item))

        clean["name"] = clean.get("name") or clean.get("item") or "Unnamed item"
        clean["id"] = clean.get("id") or str(uuid4())
        clean["quantity_multiplier"] = 1.0
        clean["serving size"] = (
            clean.get("serving size")
            or clean.get("serving_size")
            or clean.get("serving", "")
        )
        clean["confirmed"] = False
        normalized.append(clean)
    return normalized


def blank_item() -> Dict[str, Any]:
    return {
        "id": str(uuid4()),
        "name": "Flatbread (roti, custom)",
        "serving size": "1 pc",
        "protein": 0.0,
        "carbs": 0.0,
        "fiber": 0.0,
        "fats & lipids": 0.0,
        "calories": 0.0,
        "calcium": 0.0,
        "iron": 0.0,
        "potassium": 0.0,
        "vitamin C": 0.0,
        "vitamin A": 0.0,
        "vitamin D": 0.0,
        "vitamin K": 0.0,
        "vitamin B1": 0.0,
        "vitamin B6": 0.0,
        "vitamin B12": 0.0,
        "vitamin E": 0.0,
        "quantity_multiplier": 1.0,
        "confirmed": False,
    }


def _slugify_name(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower()).strip()


def prepare_prompt_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for item in items:
        try:
            quantity = float(item.get("quantity_multiplier", 1.0))
        except (TypeError, ValueError):
            quantity = 1.0
        payload.append(
            {
                "name": item.get("name", "Item"),
                "serving_size": item.get("serving size", ""),
                "quantity": quantity,
                "notes": item.get("notes", ""),
            }
        )
    return payload


def merge_nutrition_data(
    base_items: List[Dict[str, Any]], nutrition_items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    lookup = {
        _slugify_name(item.get("name")): item for item in base_items if item.get("name")
    }

    for entry in nutrition_items:
        clean = apply_field_aliases(dict(entry))

        name = clean.get("name") or clean.get("item") or "Item"
        key = _slugify_name(name)
        target = lookup.get(key)
        if not target:
            target = blank_item()
            target["name"] = name
            target["serving size"] = clean.get("serving size", target["serving size"])
            target["quantity_multiplier"] = clean.get(
                "quantity", target["quantity_multiplier"]
            )
            base_items.append(target)
            if key:
                lookup[key] = target

        for field, value in clean.items():
            if field in {"item"}:
                continue
            if field == "serving size":
                target["serving size"] = value
            elif field == "quantity":
                try:
                    target["quantity_multiplier"] = float(value)
                except (TypeError, ValueError):
                    pass
            elif field != "name":
                target[field] = value

        target["confirmed"] = True

    return base_items


def get_float(item: Dict[str, Any], key: str) -> float:
    try:
        return float(item.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def calculate_totals(items: List[Dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, float] = {key: 0.0 for key, _ in NUTRIENT_KEYS if key != "calories"}
    totals["calories"] = 0.0

    for item in items:
        qty = float(item.get("quantity_multiplier", 1.0))
        protein = get_float(item, "protein")
        carbs = get_float(item, "carbs")
        fats = get_float(item, "fats & lipids")
        macro_calories = qty * (protein * 4 + carbs * 4 + fats * 9)
        if macro_calories > 0:
            totals["calories"] += macro_calories
        else:
            totals["calories"] += qty * get_float(item, "calories")

        for key, _ in NUTRIENT_KEYS:
            if key == "calories":
                continue
            totals[key] += qty * get_float(item, key)

    return totals


def sidebar_history() -> None:
    logs = load_logs()
    st.sidebar.subheader("Logged Meals")
    if not logs:
        st.sidebar.caption("No meals logged yet.")
        return

    entries: List[Dict[str, Any]] = []
    for day, meals in logs.items():
        for meal in meals:
            entries.append({"date": day, **meal})
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

    for entry in entries[:5]:
        st.sidebar.markdown(
            f"- {entry.get('meal_type', 'Meal')} • {entry.get('date_override', entry.get('timestamp', '')[:10])}"
        )


def all_items_confirmed(items: List[Dict[str, Any]]) -> bool:
    return bool(items) and all(item.get("confirmed") for item in items)


def render_detected_items(show_nutrition: bool) -> None:
    items = st.session_state["detected_items"]
    editing_rows = st.session_state["editing_rows"]

    if not items:
        return

    with st.container():
        header_cols = st.columns([3, 1])
        header_cols[0].subheader("Detected items")
        if header_cols[1].button("➕ Add Item", key="add_item_button"):
            new_item = blank_item()
            st.session_state["detected_items"].append(new_item)
            st.session_state["editing_rows"].add(new_item["id"])
            st.rerun()

        for idx, item in enumerate(items):
            item_id = item.setdefault("id", str(uuid4()))
            item.setdefault("confirmed", False)
            cols = st.columns([3, 0.8, 0.8, 1.2])

            if item_id in editing_rows:
                new_name = cols[0].text_input(
                    "Name",
                    value=item.get("name", ""),
                    key=f"name_{item_id}",
                    label_visibility="collapsed",
                )
                new_serving = cols[0].text_input(
                    "Serving",
                    value=item.get("serving size", ""),
                    key=f"serving_{item_id}",
                    label_visibility="collapsed",
                )
                # Check if values changed and unconfirm if needed
                if new_name != item.get("name", "") or new_serving != item.get("serving size", ""):
                    item["confirmed"] = False
                    st.session_state["nutrition_ready"] = False
                item["name"] = new_name
                item["serving size"] = new_serving
                st.session_state["detected_items"][idx] = item
            else:
                cols[0].markdown(f"**{item.get('name', 'Unknown item')}**")
                cols[0].caption(item.get("serving size", ""))

            if cols[1].button("✏️", key=f"edit_{item_id}"):
                if item_id in editing_rows:
                    editing_rows.remove(item_id)
                else:
                    editing_rows.add(item_id)
                    # Unconfirm when starting to edit
                    item["confirmed"] = False
                    st.session_state["nutrition_ready"] = False
                    st.session_state["detected_items"][idx] = item
                st.rerun()

            if cols[2].button("🗑️", key=f"delete_{item_id}"):
                editing_rows.discard(item_id)
                del items[idx]
                st.rerun()

            prev_qty = float(item.get("quantity_multiplier", 1.0))
            slider_value = st.slider(
                f"Quantity for {item.get('name', '')}",
                min_value=0.25,
                max_value=5.0,
                value=prev_qty,
                step=0.25,
                key=f"qty_{item_id}",
            )
            item["quantity_multiplier"] = slider_value
            st.session_state["detected_items"][idx] = item
            if slider_value != prev_qty:
                item["confirmed"] = False
                st.session_state["nutrition_ready"] = False

            cols[3].markdown(
                "✅ Confirmed" if item.get("confirmed") else "⚠️ Needs review"
            )
            if cols[3].button(
                "Confirm",
                key=f"confirm_{item_id}",
                disabled=item.get("confirmed", False),
            ):
                item["confirmed"] = True
                editing_rows.discard(item_id)
                st.session_state["detected_items"][idx] = item
                st.rerun()

            if show_nutrition:
                nutrient_summary = ", ".join(
                    f"{label}: {slider_value * get_float(item, key):.1f}"
                    for key, label in NUTRIENT_KEYS[:3]
                )
                st.caption(nutrient_summary)
            else:
                st.caption(f"Quantity: {slider_value:.2f}×")
            st.divider()


def render_summary(items: List[Dict[str, Any]]) -> None:
    if not items:
        return

    totals = calculate_totals(items)
    st.subheader("Meal summary")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Calories", f"{totals['calories']:.0f} kcal")
    metric_cols[1].metric("Protein", f"{totals['protein']:.1f} g")
    metric_cols[2].metric("Carbs", f"{totals['carbs']:.1f} g")
    metric_cols[3].metric("Fibre", f"{totals['fiber']:.1f} g")
    metric_cols[4].metric("Fats & lipids", f"{totals['fats & lipids']:.1f} g")

    st.divider()
    st.subheader("Detailed macros & micros")

    records: List[Dict[str, Any]] = []
    for item in items:
        qty = float(item.get("quantity_multiplier", 1.0))
        row = {
            "Item": item.get("name", "Item"),
            "Serving": item.get("serving size", ""),
            "Qty (x)": qty,
        }
        for key, label in NUTRIENT_KEYS:
            value = qty * get_float(item, key)
            if key == "calories" and value == 0:
                protein = get_float(item, "protein")
                carbs = get_float(item, "carbs")
                fats = get_float(item, "fats & lipids")
                value = qty * (protein * 4 + carbs * 4 + fats * 9)
            row[label] = round(value, 2)
        records.append(row)

    if records:
        df = pd.DataFrame(records)
        st.dataframe(df, width='stretch', hide_index=True)


def render_log_page() -> None:
    ensure_state()
    st.header("Log")
    st.write("Upload a meal photo or capture one with your camera to auto-detect dishes.")

    # Initialize camera state
    if "show_camera" not in st.session_state:
        st.session_state["show_camera"] = False

    # Side by side: Browse files and Camera button
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded = st.file_uploader(
            "📁 Browse Files", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Align button with file uploader
        if st.button("📷 Camera", use_container_width=True, type="primary"):
            st.session_state["show_camera"] = not st.session_state["show_camera"]
            st.rerun()
    
    # Show camera input only when button is clicked
    camera_photo = None
    if st.session_state["show_camera"]:
        # Add custom HTML to request back camera
        st.markdown("""
        <script>
        // Request back camera when camera input is shown
        setTimeout(function() {
            const videoInputs = document.querySelectorAll('input[type="file"][accept*="image"]');
            videoInputs.forEach(input => {
                if (input.capture !== undefined) {
                    input.setAttribute('capture', 'environment');
                }
            });
        }, 100);
        </script>
        """, unsafe_allow_html=True)
        
        camera_photo = st.camera_input("Take a photo", key="camera_input")
        
        if st.button("✖ Close Camera"):
            st.session_state["show_camera"] = False
            st.rerun()

    image_data_url = None
    if camera_photo is not None:
        image_data_url = build_data_url("camera.jpg", camera_photo.getvalue())
        # Close camera after capturing
        st.session_state["show_camera"] = False
    elif uploaded is not None:
        image_data_url = build_data_url(uploaded.name, uploaded.getvalue())

    preview = image_data_url or st.session_state.get("last_image_preview")
    if preview:
        st.image(preview, caption="Selected meal photo", width='stretch')

    if st.button("Detect items", disabled=image_data_url is None):
        if not image_data_url:
            st.warning("Please upload or capture an image first.")
        else:
            with st.spinner("Analyzing meal..."):
                try:
                    result = detect_food_items(image_data_url)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to analyze image: {exc}")
                else:
                    st.session_state["raw_detection"] = result.get("text", "")
                    st.session_state["raw_nutrition"] = ""
                    st.session_state["nutrition_ready"] = False
                    st.session_state["last_image_preview"] = image_data_url
                    parsed_items = result.get("items") or []
                    if parsed_items:
                        st.session_state["detected_items"] = normalize_items(parsed_items)
                        st.session_state["editing_rows"] = set()
                        st.success("Items detected. Review and adjust quantities below.")
                    else:
                        st.warning(
                            "Could not parse structured items from the model response."
                        )

    if st.session_state["raw_detection"]:
        with st.expander("Detection response (JSON)"):
            st.code(st.session_state["raw_detection"], language="json")

    render_detected_items(show_nutrition=st.session_state["nutrition_ready"])

    meal_type = st.selectbox("Meal type", MEAL_TYPES, key="meal_type")

    confirmed = all_items_confirmed(st.session_state["detected_items"])
    if st.session_state["detected_items"] and not confirmed:
        st.info("Confirm each row before requesting macros.")

    confirm_disabled = not st.session_state["detected_items"] or not confirmed
    if st.button(
        "Confirm items & get macros",
        disabled=confirm_disabled,
    ):
        payload = prepare_prompt_items(st.session_state["detected_items"])
        with st.spinner("Generating macros and micros..."):
            try:
                nutrition = enrich_food_items(payload)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to enrich nutrition: {exc}")
            else:
                st.session_state["raw_nutrition"] = nutrition.get("text", "")
                enriched = nutrition.get("items") or []
                if enriched:
                    st.session_state["detected_items"] = merge_nutrition_data(
                        st.session_state["detected_items"], enriched
                    )
                    st.session_state["nutrition_ready"] = True
                    st.success("Macros added. Review the breakdown below.")
                else:
                    st.warning(
                        "Could not parse nutrition response. Review the raw output below."
                    )

    if st.session_state["raw_nutrition"]:
        with st.expander("Nutrition response (JSON)"):
            st.code(st.session_state["raw_nutrition"], language="json")

    if st.session_state["detected_items"]:
        if st.session_state["nutrition_ready"]:
            render_summary(st.session_state["detected_items"])
        else:
            st.info("Confirm the items first to generate macros and micros.")

    if st.session_state["nutrition_ready"] and st.session_state["detected_items"]:
        if st.button("Log meal", type="primary"):
            logs = load_logs()
            date_key = date.today().isoformat()
            entry = {
                "meal_type": meal_type,
                "timestamp": datetime.now().isoformat(),
                "items": st.session_state["detected_items"],
                "totals": calculate_totals(st.session_state["detected_items"]),
            }
            logs.setdefault(date_key, []).append(entry)
            save_logs(logs)
            st.success("Meal logged successfully.")
            st.session_state["detected_items"] = []
            st.session_state["nutrition_ready"] = False
            st.session_state["raw_detection"] = ""
            st.session_state["raw_nutrition"] = ""
            st.session_state["editing_rows"] = set()


def render_history_page() -> None:
    st.header("Logged Meals")
    logs = load_logs()
    if not logs:
        st.info("No meals logged yet.")
        return

    available_dates = sorted(logs.keys())
    default_date = date.fromisoformat(available_dates[-1])
    selected_date = st.date_input(
        "Select a date",
        value=default_date,
        min_value=date.fromisoformat(available_dates[0]),
        max_value=date.fromisoformat(available_dates[-1]),
    )

    entries = logs.get(selected_date.isoformat(), [])
    if not entries:
        st.warning("No meals logged for this date.")
        return

    for entry in entries:
        with st.container():
            st.subheader(f"{entry.get('meal_type', 'Meal')} • {selected_date.isoformat()}")
            totals = entry.get("totals", {})
            st.caption(
                f"Calories: {totals.get('calories', 0):.0f} kcal | Protein: {totals.get('protein', 0):.1f} g | Carbs: {totals.get('carbs', 0):.1f} g"
            )
            render_summary(entry.get("items", []))
            st.divider()


def main() -> None:
    st.set_page_config(page_title="Food Detection", layout="wide")
    sidebar_history()
    page = st.sidebar.radio("Pages", ["Log", "Logged meals"])

    if page == "Log":
        render_log_page()
    else:
        render_history_page()


if __name__ == "__main__":
    main()
