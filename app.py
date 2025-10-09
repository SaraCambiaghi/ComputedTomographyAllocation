from mip import Model, xsum, BINARY, CONTINUOUS, minimize
from flask import Flask, render_template, request
from catboost import CatBoostRegressor
import pandas as pd
import pickle
import math
import random
import os

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        form_data = request.form
        pazienti = []
        i = 0
        while f"pazienti[{i}][sesso]" in form_data:
            paziente = {
                "sesso": form_data[f"pazienti[{i}][sesso]"],
                "eta": int(form_data[f"pazienti[{i}][eta]"]),
                "tipo": form_data[f"pazienti[{i}][tipo]"],
                "esami": form_data.getlist(f"pazienti[{i}][esami][]"), 
                "numero_esami": int(form_data[f"pazienti[{i}][numero_esami]"])
            }

            pazienti.append(paziente)
            i += 1

        # Tutti i tipi di esami possibili (stessi del form)
        esami_possibili = ['TC CEREBRALE (SENZA MDC)','TC CEREBRALE (SENZA E CON MDC)',
        'TC CRANIO (SENZA MDC) SELLA TURCICA, ORBITE', 'TC MASSICCIO FACCIALE (SENZA MDC)',
        'ANGIO-TC INTRACRANICO', 'TC COLONNA CERVICALE (SENZA MDC)', 'TC COLONNA DORSALE (SENZA MDC)', 
        'TC COLONNA LOMBO-SACRALE (SENZA MDC)', 'TC COLLO (SENZA E CON MDC)', 'ANGIO-TC TRONCHI SOVRAORTICI', 
        'TC TORACE (SENZA MDC)', 'TC TORACE (SENZA E CON MDC)', 'TC ADDOME COMPLETO (SENZA MDC)', 
        'TC ADDOME COMPLETO (SENZA E CON MDC)', 'TC BACINO E ART. SACRO-ILIACHE (SENZA MDC)', 
        'TC GOMITO/AVAMBRACCIO DX (SENZA MDC)','ANGIO-TC ARTO INFERIORE DX', 'ANGIO-TC ARTO INFERIORE SX', 
        'OTHER_EXAMS']

        # Crea il DataFrame
        rows = []
        for p in pazienti:
            row = {
                "gender": int(p["sesso"]),
                "Body/Neuro": int(p["tipo"]),
                "age": int(p["eta"]),
                "exams_number": int(p["numero_esami"])
            }
            for esame in esami_possibili:
                row[esame] = 1 if esame in p["esami"] else 0
            rows.append(row)
        df = pd.DataFrame(rows)
        df = df.reset_index(drop=True)
        n_pat = len(df)

        criterio = request.form.get("criterioDurata")
        
        ## ESAMI 
        def round_robin_exam(predizioni, tipo):
            n_pat = len(predizioni)
            round_robin_assignments = []
            costs_body = [0,0,0,0,0]
            costs_neuro = [0,0,0,0,0]
            day = 0
            for pat in range(n_pat):
                # assegno il paziente al giorno
                round_robin_assignments.append((pat,day))
                # aggiungo occupazione a seconda del tipo
                if tipo[pat] == 1: 
                    costs_body[day] += predizioni[pat]
                elif tipo[pat] == 0:
                    costs_neuro[day] += predizioni[pat]
                # passo al giorno successivo
                day += 1
                # se finisce la settimana, ricomincio
                if day >= 5:
                    day = 0  
            # calcolo occupazione totale body + neuro
            costs = [a + b for a, b in zip(costs_neuro, costs_body)]
            # ritorno il massimo tra i giorni e gli assignments
            return  {
                "z": max(costs),
                "assignments": round_robin_assignments,
                "occupazione_total": costs
            }

        def opt_model_exam(predizioni, tipo, m = 5):
            n = len(predizioni)  # pazienti
            model = Model()

            # x[i][j] = 1 se paziente i va nel giorno j
            x = [[model.add_var(var_type=BINARY) for j in range(m)] for i in range(n)]

            # Occupazione giornaliera Body e Neuro
            zb = [model.add_var(var_type=CONTINUOUS) for j in range(m)]
            zn = [model.add_var(var_type=CONTINUOUS) for j in range(m)]

            # Variabile occupazione massima complessiva
            z = model.add_var(var_type=CONTINUOUS)

            # Ogni paziente in un solo giorno
            for i in range(n):
                model += xsum(x[i][j] for j in range(m)) == 1

            # Vincoli per calcolare occupazioni giornaliere
            for j in range(m):
                # Body
                model += zb[j] == xsum(tipo[i] * predizioni[i] * x[i][j] for i in range(n))
                # Neuro
                model += zn[j] == xsum((1 - tipo[i]) * predizioni[i] * x[i][j] for i in range(n))
                # z >= occupazione totale per quel giorno
                model += z >= zb[j] + zn[j]

            # Minimizzo occupazione max + penalità totale
            model.objective = minimize(z) 

            model.optimize(max_seconds=300)

            assignments = []
            for j in range(m):
                for i in range(n):
                    if x[i][j].x >= 0.99:
                        assignments.append((i, j))
            
            occupazione_body = [zb[j].x for j in range(m)]
            occupazione_neuro = [zn[j].x for j in range(m)]
            occupazione_total = [b + n for b, n in zip(occupazione_body, occupazione_neuro)]

            return {
                "z": z.x,
                "assignments": assignments,
                "occupazione_total": occupazione_total
            }

        def compute_real_exam(d_real, tipo, result, random_result):
            employ_rand_body = [0,0,0,0,0]
            employ_opt_body = [0,0,0,0,0]
            employ_rand_neuro = [0,0,0,0,0]
            employ_opt_neuro = [0,0,0,0,0]
            employ_rand = [0,0,0,0,0]
            employ_opt = [0,0,0,0,0]
            for p in result['assignments']:
                if tipo[p[0]] == 0:
                    employ_opt_neuro[p[1]] += d_real[p[0]]
                elif tipo[p[0]] == 1:
                    employ_opt_body[p[1]] += d_real[p[0]]
                employ_opt[p[1]] += d_real[p[0]]
            for p in random_result['assignments']:
                if tipo[p[0]] == 0:
                    employ_rand_neuro[p[1]] += d_real[p[0]]
                elif tipo[p[0]] == 1:
                    employ_rand_body[p[1]] += d_real[p[0]]
                employ_rand[p[1]] += d_real[p[0]]
            return max(employ_opt), max(employ_rand)
            
        ## REFERTAZIONE
        def round_robin_report(predizioni, tipo):
            n_pat = len(predizioni)
            round_robin_assignments = []
            costs_body = [0,0,0,0,0]
            costs_neuro = [0,0,0,0,0]
            day = 0
            for pat in range(n_pat):
                round_robin_assignments.append((pat,day))
                if tipo[pat] == 1: 
                    costs_body[day] += predizioni[pat]
                elif tipo[pat] == 0:
                    costs_neuro[day] += predizioni[pat]
                day += 1
                if day >= 5:
                    day = 0  
            costs = [max(a, b) for a, b in zip(costs_body, costs_neuro)]
            return  {
                "z": max(costs),
                "assignments": round_robin_assignments,
                "occupazione_total": costs
            }

        def opt_model_report(predizioni, tipo, m = 5):
            n = len(predizioni)  # pazienti
            model = Model()

            # x[i][j] = 1 se paziente i va nel giorno j
            x = [[model.add_var(var_type=BINARY) for j in range(m)] for i in range(n)]

            # Occupazione giornaliera Body e Neuro
            zb = [model.add_var(var_type=CONTINUOUS) for j in range(m)]
            zn = [model.add_var(var_type=CONTINUOUS) for j in range(m)]

            # Variabile occupazione massima complessiva
            z = model.add_var(var_type=CONTINUOUS)

            # Ogni paziente in un solo giorno
            for i in range(n):
                model += xsum(x[i][j] for j in range(m)) == 1

            # Vincoli per calcolare occupazioni giornaliere
            for j in range(m):
                # Body
                model += zb[j] == xsum(tipo[i] * predizioni[i] * x[i][j] for i in range(n))
                # Neuro
                model += zn[j] == xsum((1 - tipo[i]) * predizioni[i] * x[i][j] for i in range(n))
                # z = massimo tra body e neuro di ogni giorno
                model += z >= zn[j]
                model += z >= zb[j]


            # Minimizzo occupazione max + penalità totale
            model.objective = minimize(z) 

            model.optimize(max_seconds=300)

            assignments = []
            for j in range(m):
                for i in range(n):
                    if x[i][j].x >= 0.99:
                        assignments.append((i, j))


            occupazione_body = [zb[j].x for j in range(m)]
            occupazione_neuro = [zn[j].x for j in range(m)]
            occupazione_total = [max(b, n) for b, n in zip(occupazione_body, occupazione_neuro)]

            return {
                "z": z.x,
                "assignments": assignments,
                "occupazione_total": occupazione_total
            }

        ## OCCUPAZIONI CON DURATE REALI
        def compute_real_report(d_real, tipo, result, random_result):
            employ_rand_body = [0,0,0,0,0]
            employ_opt_body = [0,0,0,0,0]
            employ_rand_neuro = [0,0,0,0,0]
            employ_opt_neuro = [0,0,0,0,0]
            for p in result['assignments']:
                if tipo[p[0]] == 0:
                    employ_opt_neuro[p[1]] += d_real[p[0]]
                elif tipo[p[0]] == 1:
                    employ_opt_body[p[1]] += d_real[p[0]]
            for p in random_result['assignments']:
                if tipo[p[0]] == 0:
                    employ_rand_neuro[p[1]] += d_real[p[0]]
                elif tipo[p[0]] == 1:
                    employ_rand_body[p[1]] += d_real[p[0]]
            return max(max(employ_opt_body), max(employ_opt_neuro)), max(max(employ_rand_body), max(employ_rand_neuro))

        ## calcolo durate ""reali""
        with open('residui_exam.pkl', 'rb') as file:
            residui_exam = pickle.load(file)
        with open('residui_reporting.pkl', 'rb') as file:
            residui_reporting = pickle.load(file)
        
        # aggiungo i residui
        def compute_d_real(residui, predizioni):
            d_real = []
            residui_list = residui.tolist()
            for i in range(len(predizioni)):
                d_real.append(predizioni[i]+random.choice(residui_list))
            return d_real


        # calcolo occupazione con durate reali
      
        # Inizializza le variabili prima del controllo del criterio
        predizioni = []
        d_real = []
        result = None
        random_result = None
        rand_body = 0
        rand_neuro = 0
        opt_body = 0
        opt_neuro = 0
            
        if criterio == "esami":
            # carico il modello e sistemo le colonne
            cat_model = CatBoostRegressor()
            cat_model.load_model("catboost_model_ex.cbm")
            missing_cols = set(cat_model.feature_names_) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            df_model = df[cat_model.feature_names_]  # Crea una copia per il modello
            # predico e arrotondo
            predizioni_reg = cat_model.predict(df_model)
            predizioni = [math.ceil(p) for p in predizioni_reg]
            # calcolo i due schedule 
            result = opt_model_exam(predizioni, df['Body/Neuro'].tolist()) # modello
            random_result = round_robin_exam(predizioni, df['Body/Neuro'].tolist()) # baseline
            # iniziazzo workload di ogni giorni
            val_opt = 0
            val_rr = 0
            # simulo 100 scenari
            for o in range(100):
                d_real = compute_d_real(residui_exam, predizioni)
                employ_opt, employ_rand = compute_real_exam(d_real, df['Body/Neuro'].tolist(), result, random_result)
                val_opt += employ_opt
                val_rr += employ_rand

        elif criterio == "refertazione":
            # carico il modello e sistemo le colonne
            cat_model = CatBoostRegressor()
            cat_model.load_model("catboost_model_rep.cbm")
            missing_cols = set(cat_model.feature_names_) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            # predico e arrotondo
            df_model = df[cat_model.feature_names_]  # Crea una copia per il modello
            predizioni_reg = cat_model.predict(df_model)
            predizioni = [math.ceil(p) for p in predizioni_reg]
            # calcolo i due schedule
            result = opt_model_report(predizioni, df['Body/Neuro'].tolist())
            # result = opt_euristic(predizioni, df['Body/Neuro'].tolist())
            random_result = round_robin_report(predizioni, df['Body/Neuro'].tolist())
            # iniziazzo workload di ogni giorni
            val_opt = 0
            val_rr = 0
            # simulo 100 scenari
            for o in range(100):
                d_real = compute_d_real(residui_reporting, predizioni)
                employ_opt, employ_rand = compute_real_report(d_real, df['Body/Neuro'].tolist(), result, random_result)
                val_opt += employ_opt
                val_rr += employ_rand

        return render_template(
            "conferma.html",
            pazienti=pazienti,
            table=df.to_html(classes="table"),
            predizioni=predizioni,
            real_durations=d_real,
            result=result,
            random_result=random_result,
            employ_old = val_rr / 100, 
            employ_opt = val_opt / 100
        )

    return render_template("index.html")

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))