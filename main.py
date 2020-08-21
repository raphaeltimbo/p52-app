import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import ccp
import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from itertools import cycle
from pathlib import Path

colors = ccp.plotly_theme.tableau_colors
Q_ = ccp.Q_

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])

# ----------------------
# Data
# ----------------------

cases = {
    "LP FD": {
        "composition": {
            "methane": 70.593,
            "ethane": 12.219,
            "propane": 8.049,
            "ibutane": 0.49,
            "nbutane": 3.56,
            "ipentane": 0.28,
            "npentane": 1.36,
            "hexane": 0.84,
            "heptane": 0.48,
            "octane": 0.12,
            "nonane": 0.02,
            "n2": 0.57,
            "co2": 0.6,
        },
        "path": "01_LPFD_Filtered/casesResults",
    }
}

# ----------------------
# Controls
# ----------------------


def controls(n):
    case_labels = ["A", "B"]
    c = dbc.Card(
        [
            dbc.FormGroup(
                [
                    dbc.Label(f"Caso {case_labels[n]}"),
                    dcc.Dropdown(
                        id=f"case{n}",
                        options=[{"label": c, "value": c} for c in cases.keys()],
                        value=list(cases.keys())[0],
                    ),
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Pressão de Extração"),
                    dbc.Input(id=f"p_ext{n}", type="number", value=2000),
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Queda de Temperatura na Tubulação"),
                    dbc.Input(id=f"T_cool{n}", type="number", value=10),
                ]
            ),
        ],
        body=True,
    )

    return c


temperature_units = ["degC", "degK"]
pressure_units = ["kPa", "MPa", "bar"]
reference_pressure = ["abs"]

units_control = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label("Unidades de Temperatura"),
                            dcc.Dropdown(
                                id="T_units",
                                options=[
                                    {"label": c, "value": c} for c in temperature_units
                                ],
                                value="degC",
                            ),
                        ]
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label("Unidades de Pressão"),
                            dcc.Dropdown(
                                id="p_units",
                                options=[
                                    {"label": c, "value": c} for c in pressure_units
                                ],
                                value="kPa",
                            ),
                        ]
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Label("Pressão de Referência"),
                            dcc.Dropdown(
                                id="p_reference",
                                options=[
                                    {"label": c, "value": c} for c in reference_pressure
                                ],
                                value="abs",
                            ),
                        ]
                    ),
                    md=4,
                ),
            ]
        ),
    ],
    body=True,
)


def figures(n):
    fig = dbc.Col(dcc.Graph(id=f"fig{n}"), md=4)

    return fig


# ----------------------
# App Layout
# ----------------------

app.layout = dbc.Container(
    [
        html.H1("P52"),
        html.Hr(),
        dbc.Row([dbc.Col(units_control, md=8)]),
        dbc.Row([dbc.Col(controls(0), md=4), dbc.Col(controls(1), md=4)]),
        dbc.Row([figures(0), figures(1)]),
        dbc.Row([dbc.Col(dbc.Table(id="results_table"), md=8)]),
    ],
    fluid=True,
)

# ----------------------
# Callbacks
# ----------------------


def run_analysis(path, comp, p_ext, T_cool, T_units, p_units, caso=None):
    """Run analysis.

    Parameters
    ----------
    path : str
        Path with results from CFD.
    comp : dict
        Fluid composition.
    p_ext : pint.Quantity
        Extraction pressure (abs).

    Returns
    -------
    fig, descr
        Figure and results description.
    """

    # dummy state to create the envelope
    dummy_state = ccp.State.define(p=1e6, T=300, fluid=comp)
    fig = dummy_state.plot_envelope(T_units=T_units, p_units=p_units)

    colors_plot = colors.copy()
    colors_plot.pop("blue")
    colors_plot.pop("orange")
    colors_plot = cycle(colors_plot.values())

    results_path = Path(path)
    results_files = sorted(list(results_path.glob("*")))

    compressor_list = []
    condition_list = []
    gap_list = []
    T_inj_list = []
    T_h_list = []

    for file in results_files:
        data = {}
        hull = {"x": [], "y": []}
        record_hull = False
        with open(file, "r") as f:
            lines = f.readlines()
            for l in lines:
                try:
                    k, v = l.split(" ")
                    data[k] = float(v)
                except ValueError:
                    pass

                if "caseHull" in l:
                    record_hull = True
                if "}" in l:
                    record_hull = False

                if record_hull:
                    try:
                        x, y = map(float, l.split(" "))
                        hull["x"].append(x)
                        hull["y"].append(y)
                    except ValueError:
                        pass

        # plot cfd results
        color = next(colors_plot)
        hull_T = Q_(hull["x"], "degK").to(T_units).m
        hull_p = Q_(hull["y"], "Pa").to(p_units).m
        T_inj = Q_(data["Temperatura"], "degK").to(T_units).m
        p_inj = Q_(data["Pressao_entrada"], "Pa").to(p_units).m
        name = f"{data['h'] * 1e6} μm, Tᵢ = {T_inj:.2f} {T_units}"
        hovertemplate = (
            f"Temperature ({T_units}): %{{x}}<br>" f"Pressure ({p_units}): %{{y}}"
        )
        fig.add_trace(
            go.Scatter(
                x=hull_T,
                y=hull_p,
                hovertemplate=hovertemplate,
                showlegend=True,
                name=name,
                line=dict(color=color),
            )
        )

        # calculate T_heater
        # define state
        s = ccp.State.define(
            p=data["Pressao_entrada"], T=data["Temperatura"], fluid=comp
        )
        # isoenthalpic process
        s.update(p=p_ext, h=s.h())
        s.update(p=s.p(), T=(s.T().m + T_cool))
        T_h = s.T().to(T_units).m
        name_heater = f"{data['h'] * 1e6} μm, Tₕ = {T_h:.2f} {T_units}"
        fig = s.plot_point(
            T_units=T_units,
            p_units=p_units,
            fig=fig,
            showlegend=True,
            name=name_heater,
            mode="markers",
            marker=dict(color=color),
        )

        compressor, condition = caso.split(" ")
        compressor_list.append(compressor)
        condition_list.append(condition)
        T_inj_list.append(round(T_inj, 2))
        T_h_list.append(round(T_h, 2))
        gap_list.append(data["h"] * 1e6)

    descr = f"""
    Caso analisado:               {caso}
    Composição:                   {comp}
    Pressão de extração:          {p_ext:.2f}
    Pressão de injeção:           {p_inj:.2f}
    """

    atm = Q_(1, "atm").to(p_units).m
    df = pd.DataFrame(
        {
            "Parâmetros": [
                f"Pressão de Extração ({p_units})",
                f"Pressão de Injeção ({p_units})",
            ],
            "Valores (abs)": [p_ext.to(p_units).m, p_inj],
            "Valores (gauge)": [p_ext.to(p_units).m - atm, p_inj - atm],
        }
    )
    df = df.set_index("Parâmetros")

    dfr = pd.DataFrame(
        {
            "Compressor": compressor_list,
            "Condição Operacional": condition_list,
            "h (μm)": gap_list,
            f"Tᵢ {T_units}": T_inj_list,
            f"Tₕ {T_units}": T_h_list,
        },
    )
    dfr = dfr.set_index("Compressor")

    return fig, descr, df, dfr


@app.callback(
    [
        Output(component_id="fig0", component_property="figure"),
        Output(component_id="fig1", component_property="figure"),
        Output(component_id="results_table", component_property="children"),
    ],
    [
        Input(component_id="case0", component_property="value"),
        Input(component_id="p_ext0", component_property="value"),
        Input(component_id="T_cool0", component_property="value"),
        Input(component_id="case1", component_property="value"),
        Input(component_id="p_ext1", component_property="value"),
        Input(component_id="T_cool1", component_property="value"),
        Input(component_id="T_units", component_property="value"),
        Input(component_id="p_units", component_property="value"),
    ],
)
def update_results(case0, p_ext0, T_cool0, case1, p_ext1, T_cool1, T_units, p_units):
    """App function.

    Parameters
    ----------
    case : str
    p_ext : float
    T_cool : float


    """

    path_p52 = (
        Path(__file__).parent
        / "Filtrados"
    )

    p_ext0 = Q_(p_ext0, p_units)
    case_data0 = cases[case0]
    path0 = path_p52 / case_data0["path"]
    comp0 = case_data0["composition"]

    fig0, descr0, df0, dfr0 = run_analysis(
        path0, comp0, p_ext0, T_cool0, T_units, p_units, caso=case0
    )

    p_ext1 = Q_(p_ext1, p_units)
    case_data1 = cases[case1]
    path1 = path_p52 / case_data1["path"]
    comp1 = case_data1["composition"]

    fig1, descr1, df1, dfr1 = run_analysis(
        path1, comp1, p_ext1, T_cool1, T_units, p_units, caso=case1
    )

    df = dfr0.drop(f"Tᵢ {T_units}", 1)
    df = df.rename(columns={f"Tₕ {T_units}": f"Tₕ {T_units} - A"})
    df[f"Tₕ {T_units} - B"] = dfr1[f"Tₕ {T_units}"]
    df["Delta (A - B)"] = df[f"Tₕ {T_units} - A"] - df[f"Tₕ {T_units} - B"]
    df = df.reset_index(drop=False)
    results_table = dbc.Table.from_dataframe(
        df, striped=True, bordered=True, hover=True
    )

    return fig0, fig1, results_table


if __name__ == "__main__":
    # debug mode
    # app.run_server(debug=True, port=5006)
    # production
    app.run_server(host="0.0.0.0", port=5006)
