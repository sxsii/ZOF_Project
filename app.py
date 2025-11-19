from flask import Flask, render_template, request, redirect, url_for
import sympy as sp
import numpy as np

app = Flask(__name__)

def make_function(expr_str):
    x = sp.symbols('x')
    expr = sp.sympify(expr_str)
    f = sp.lambdify(x, expr, 'numpy')
    df = sp.lambdify(x, sp.diff(expr, x), 'numpy')
    return f, df

# Reuse implementations (slightly adapted) for web results
def secant(f, x0, x1, tol, max_iter):
    rows=[]
    for i in range(1, max_iter+1):
        f0 = float(f(x0)); f1 = float(f(x1))
        if f1 - f0 == 0:
            raise ZeroDivisionError("Division by zero in secant.")
        x2 = x1 - f1*(x1-x0)/(f1-f0)
        err = abs(x2 - x1)
        rows.append({'iter':i, 'x0':x0, 'x1':x1, 'x2':x2, 'f0':f0, 'f1':f1, 'err':err})
        if abs(f(x2)) < tol or err < tol:
            return x2, err, i, rows
        x0, x1 = x1, x2
    return x2, err, max_iter, rows

def newton_raphson(f, df, x0, tol, max_iter):
    rows=[]
    x = x0
    for i in range(1, max_iter+1):
        fx = float(f(x)); dfx = float(df(x))
        if dfx == 0:
            raise ZeroDivisionError("Zero derivative.")
        x_new = x - fx/dfx
        err = abs(x_new - x)
        rows.append({'iter':i, 'x':x, 'fx':fx, 'dfx':dfx, 'xnew':x_new, 'err':err})
        if abs(fx) < tol or err < tol:
            return x_new, err, i, rows
        x = x_new
    return x, err, max_iter, rows

# Add bisection, regula falsi, fixed point, modified secant analogously
def bisection(f, a, b, tol, max_iter):
    if f(a)*f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    rows=[]
    fa = float(f(a)); fb = float(f(b))
    for i in range(1, max_iter+1):
        c = (a+b)/2.0
        fc = float(f(c))
        err = abs(b-a)/2.0
        rows.append({'iter':i,'a':a,'b':b,'c':c,'fa':fa,'fb':fb,'fc':fc,'err':err})
        if abs(fc) < tol or err < tol:
            return c, err, i, rows
        if fa*fc < 0:
            b = c; fb = fc
        else:
            a = c; fa = fc
    return c, err, max_iter, rows

def regula_falsi(f, a, b, tol, max_iter):
    if f(a)*f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    rows=[]
    fa = float(f(a)); fb = float(f(b))
    x_prev = None
    for i in range(1, max_iter+1):
        c = (a*fb - b*fa)/(fb - fa)
        fc = float(f(c))
        err = abs(c - (x_prev if x_prev is not None else c))
        rows.append({'iter':i,'a':a,'b':b,'c':c,'fa':fa,'fb':fb,'fc':fc,'err':err})
        if abs(fc) < tol or err < tol:
            return c, err, i, rows
        if fa*fc < 0:
            b = c; fb = fc
        else:
            a = c; fa = fc
        x_prev = c
    return c, err, max_iter, rows

def fixed_point(g_func, x0, tol, max_iter):
    rows=[]
    x=x0
    for i in range(1, max_iter+1):
        x_new = float(g_func(x))
        err = abs(x_new - x)
        rows.append({'iter':i,'x':x,'xnew':x_new,'err':err})
        if err < tol:
            return x_new, err, i, rows
        x = x_new
    return x, err, max_iter, rows

def modified_secant(f, x0, delta_frac, tol, max_iter):
    rows=[]
    x = x0
    for i in range(1, max_iter+1):
        delta = delta_frac * (abs(x) if x!=0 else 1.0)
        f_x = float(f(x))
        f_xd = float(f(x+delta))
        denom = f_xd - f_x
        if denom == 0:
            raise ZeroDivisionError("Zero denominator in modified secant.")
        x_new = x - f_x * (delta) / denom
        err = abs(x_new - x)
        rows.append({'iter':i,'x':x,'delta':delta,'fx':f_x,'fxd':f_xd,'xnew':x_new,'err':err})
        if abs(f_x) < tol or err < tol:
            return x_new, err, i, rows
        x = x_new
    return x, err, max_iter, rows

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        f_str = request.form['f']
        method = request.form['method']
        tol = float(request.form.get('tol',1e-6))
        max_iter = int(request.form.get('max_iter',50))
        try:
            f, df = make_function(f_str)
        except Exception as e:
            return render_template('index.html', error=f"Error parsing f(x): {e}")
        try:
            if method == 'bisection':
                a = float(request.form['a']); b = float(request.form['b'])
                root, err, iters, rows = bisection(f, a, b, tol, max_iter)
            elif method == 'regula':
                a = float(request.form['a']); b = float(request.form['b'])
                root, err, iters, rows = regula_falsi(f, a, b, tol, max_iter)
            elif method == 'secant':
                x0 = float(request.form['x0']); x1 = float(request.form['x1'])
                root, err, iters, rows = secant(f, x0, x1, tol, max_iter)
            elif method == 'newton':
                x0 = float(request.form['x0'])
                root, err, iters, rows = newton_raphson(f, df, x0, tol, max_iter)
            elif method == 'fixed':
                gpresent = request.form.get('g_func','').strip()
                if gpresent:
                    x = sp.symbols('x')
                    g_expr = sp.sympify(gpresent)
                    g_func = sp.lambdify(x, g_expr, 'numpy')
                else:
                    lam = float(request.form.get('lambda_val', 0.1))
                    g_func = lambda x: x - lam * f(x)
                x0 = float(request.form['x0'])
                root, err, iters, rows = fixed_point(g_func, x0, tol, max_iter)
            elif method == 'modified_secant':
                x0 = float(request.form['x0'])
                delta = float(request.form.get('delta',1e-3))
                root, err, iters, rows = modified_secant(f, x0, delta, tol, max_iter)
            else:
                return render_template('index.html', error="Unknown method")
        except Exception as e:
            return render_template('index.html', error=f"Computation error: {e}")
        return render_template('index.html', result=True, root=root, err=err, iters=iters, rows=rows, method=method, f_str=f_str)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
