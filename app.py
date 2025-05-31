from pyomo.environ import *
from flask import Flask, render_template, request
from catboost import CatBoostRegressor
import pandas as pd
import os
import pickle
from sklearn.metrics.pairwise import nan_euclidean_distances
import math
import random

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        pazienti = []
        i = 0
        while f"pazienti[{i}][sesso]" in request.form:
            paziente = {
                "sesso": request.form.get(f"pazienti[{i}][sesso]"),
                "dipartimento": request.form.get(f"pazienti[{i}][dipartimento]"),
                "tipo": request.form.get(f"pazienti[{i}][tipo]"),
                "eta": int(request.form.get(f"pazienti[{i}][eta]")),
                "numero_esami": int(request.form.get(f"pazienti[{i}][numero_esami]")),
                "esami": request.form.getlist(f"pazienti[{i}][esami][]")
            }
            pazienti.append(paziente)
            i += 1

        esami_possibili = [
       'TC CEREBRALE (SENZA MDC)', 'TC COLONNA LOMBO-SACRALE (SENZA MDC) ',
       'TC COLONNA CERVICALE (SENZA MDC)  ',
       'TC CRANIO (SENZA MDC)  _SELLA TURCICA, ORBITE_',
       'TC TORACE (SENZA MDC)', 'TC GOMITO/AVAMBRACCIO DX (SENZA MDC)',
       'TC MASSICCIO FACCIALE (SENZA MDC)',
       'TC ADDOME COMPLETO (SENZA E CON MDC)', 'ANGIO-TC TRONCHI SOVRAORTICI',
       'ANGIO-TC INTRACRANICO', 'TC BACINO E ART. SACRO-ILIACHE (SENZA MDC)',
       'ANGIO-TC ARTO INFERIORE DX', 'ANGIO-TC ARTO INFERIORE SX',
       'TC CEREBRALE (SENZA E CON MDC)', 'TC TORACE (SENZA E CON MDC)',
       'TC ADDOME COMPLETO (SENZA MDC)', 'TC COLLO (SENZA E CON MDC) ',
       'TC COLONNA DORSALE (SENZA MDC) ', 'OTHER_EXAMS'
        ]

        rows = []
        for p in pazienti:
            row = {
                "gender": p["sesso"],
                "department": p["dipartimento"],
                "Body/Neuro": p["tipo"],
                "age": p["eta"],
                "exams_number": p["numero_esami"]
            }
            for esame in esami_possibili:
                row[esame] = 1 if esame in p["esami"] else 0
            rows.append(row)
        df = pd.DataFrame(rows)
        df = df.reset_index(drop=True)
        n_pat = len(df)

        criterio = request.form.get("criterioDurata")
        
        def opt_model(predizioni):
            d = list(predizioni)
            I = list(range(len(d)))
            J = [0, 1, 2, 3, 4]  

            model = ConcreteModel()
            model.I = Set(initialize=I)
            model.J = Set(initialize=J)
            model.x = Var(model.I, model.J, domain=Binary)
            model.z = Var(domain=NonNegativeReals)

            def schedule_once(m, i):
                return sum(m.x[i, j] for j in m.J) == 1
            model.assignment = Constraint(model.I, rule=schedule_once)

            def set_z(m, j):
                return sum(d[i] * m.x[i, j] for i in m.I) <= m.z
            model.load_constraints = Constraint(model.J, rule=set_z)

            model.obj = Objective(expr=model.z, sense=minimize)

            solver = SolverFactory("cbc", executable="./cbc")
            # solver = SolverFactory("cbc", executable="./cbc.exe")
            solver.solve(model)

            assignments = []
            for j in model.J:
                for i in model.I:
                    if model.x[i, j]() >= 0.5:
                        assignments.append((i, j))
            
            result = {
                "z": model.z(),
                "assignments": assignments,
            }

            return result

        def random_model(m, n_pat):
            random_assignments = []
            costs = [0,0,0,0,0]
            day = 0
            for pat in range(n_pat):
                random_assignments.append((pat,day))
                costs[day] += m
                day += 1
                if day >= 5:
                    day = 0  

            random_result = {
                "z": max(costs),
                "assignments": random_assignments,
            }

            return random_result

        with open('residui_exam.pkl', 'rb') as file:
            residui_exam = pickle.load(file)
        with open('residui_reporting.pkl', 'rb') as file:
            residui_reporting = pickle.load(file)
        

        def compute_real(d_real, result, random_result):
            employ_old = [0,0,0,0,0]
            employ_opt = [0,0,0,0,0]
            for p in result['assignments']:
                employ_opt[p[1]] += d_real[p[0]]
            for p in random_result['assignments']:
                employ_old[p[1]] += d_real[p[0]]
            return employ_old, employ_opt
        
        def compute_d_real(residui, predizioni):
            d_real = []
            for i in range(len(predizioni)):
                d_real.append(predizioni[i]+random.choice(residui))
            return d_real
        

            
        if criterio == "esami":
            mean_old = 0 
            mean_opt = 0
            for o in range(100):
                cat_model = CatBoostRegressor()
                cat_model.load_model("catboost_model.cbm")
                df = df[cat_model.feature_names_]
                predizioni1 = cat_model.predict(df)
                predizioni = [math.ceil(p) for p in predizioni1]
                result = opt_model(predizioni)
                random_result = random_model(11, n_pat)
                d_real = compute_d_real(residui_exam, predizioni)
                employ_old, employ_opt = compute_real(d_real, result, random_result)
                mean_old += max(employ_old)
                mean_opt += max(employ_opt)

        if criterio == "refertazione":
            mean_old = 0 
            mean_opt = 0
            for o in range(100):
                cat_model = CatBoostRegressor()
                cat_model.load_model("catboost_model_rep.cbm")
                df = df[cat_model.feature_names_]
                predizioni1 = cat_model.predict(df)
                predizioni = [math.ceil(p) for p in predizioni1]
                result = opt_model(predizioni)
                random_result = random_model(30.6, n_pat)
                d_real = compute_d_real(residui_reporting, predizioni)
                employ_old, employ_opt = compute_real(d_real, result, random_result)
                mean_old += max(employ_old)
                mean_opt += max(employ_opt)

        return render_template(
            "conferma.html",
            pazienti=pazienti,
            table=df.to_html(classes="table"),
            predizioni=predizioni,
            real_durations=d_real,
            result=result,
            random_result=random_result,
            employ_old=mean_old/100,
            employ_opt=mean_opt/100
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
