#!/usr/bin/env python3
"""Modal dialog for choosing the efficient-frontier forecast policy.

The chosen policy is cached process-globally so downstream consumers (e.g. the
portfolio builder) can read it without re-prompting the user.
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QRadioButton,
                             QButtonGroup, QPushButton, QHBoxLayout)

from ticker import FORECAST_POLICIES, PORTFOLIO_CONFIG


# Set by ask_forecast_policy() once the user makes a choice; read via
# get_forecast_policy(). None means "never asked this session".
_cached_policy = None


def _config_default():
    return PORTFOLIO_CONFIG.get("forecast_policy", "past")


def get_forecast_policy():
    """Return the session's chosen policy, or the config default if never asked."""
    return _cached_policy or _config_default()


def ask_forecast_policy(default_policy=None):
    """Modal dialog letting the user pick the EF forecast policy.
    Caches and returns the chosen policy id; returns the default if cancelled.
    default_policy falls back to the last cached choice, then the config value."""
    global _cached_policy
    default = default_policy or get_forecast_policy()

    dialog = QDialog()
    dialog.setWindowTitle("Forecast Policy")
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel("Hi again, Is this forecast policy ok with you?"))

    group = QButtonGroup(dialog)
    buttons = {}
    for policy_id, label in FORECAST_POLICIES.items():
        btn = QRadioButton(label)
        group.addButton(btn)
        layout.addWidget(btn)
        buttons[policy_id] = btn
    buttons.get(default, next(iter(buttons.values()))).setChecked(True)

    btn_row = QHBoxLayout()
    btn_row.addStretch()
    ok_btn = QPushButton("OK")
    ok_btn.clicked.connect(dialog.accept)
    btn_row.addWidget(ok_btn)
    layout.addLayout(btn_row)

    if dialog.exec_() != QDialog.Accepted:
        _cached_policy = default
        return default
    for policy_id, btn in buttons.items():
        if btn.isChecked():
            _cached_policy = policy_id
            return policy_id
    _cached_policy = default
    return default
