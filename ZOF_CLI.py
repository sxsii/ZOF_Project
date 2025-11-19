#!/usr/bin/env python3
"""
ZOF_CLI.py
Interactive CLI for zero-of-functions solver implementing:
Bisection, Regula Falsi, Secant, Newton-Raphson, Fixed Point, Modified Secant.
Requires: sympy, numpy
"""

import sys
from math import isfinite
import numpy as np
import sympy as sp

def make_function(expr_str):
    x = sp.symbols('x')
    try:
        expr = sp.sympify(expr_str)
    except Exception as e:
        raise ValueError(f"Couldn't parse function: {e}")
    f = sp.lambdify(x, expr, 'numpy')
    df_expr = sp.diff(expr, x)
    df = sp.lambdify(x, df_expr, 'numpy')
    return f, df, expr, df_expr

def print_row(cols, widths):
    line = " | ".join(str(c).ljust(w) for c,w in zip(cols,widths))
    print(line)

def bisection(f, a, b, tol, max_iter):
    if f(a)*f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    rows=[]
    fa = f(a); fb = f(b)
    for i in range(1, max_iter+1):
        c = (a+b)/2.0
        fc = f(c)
        err = abs(b-a)/2.0
        rows.append((i, a, b, c, fa, fb, fc, err))
        if abs(fc) < tol or err < tol:
            return c, err, i, rows
        if fa*fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c, err, max_iter, rows

def regula_falsi(f, a, b, tol, max_iter):
    if f(a)*f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    rows=[]
    fa = f(a); fb = f(b)
    x_prev = None
    for i in range(1, max_iter+1):
        c = (a*fb - b*fa)/(fb - fa)
        fc = f(c)
        err = abs(c - (x_prev if x_prev is not None else c))
        rows.append((i, a, b, c, fa, fb, fc, err))
        if abs(fc) < tol or err < tol:
            return c, err, i, rows
        if fa*fc < 0:
            b = c; fb = fc
        else:
            a = c; fa = fc
        x_prev = c
    return c, err, max_iter, rows

def secant(f, x0, x1, tol, max_iter):
    rows=[]
    for i in range(1, max_iter+1):
        f0 = f(x0); f1 = f(x1)
        if f1 - f0 == 0:
            raise ZeroDivisionError("Division by zero in secant step.")
        x2 = x1 - f1*(x1-x0)/(f1-f0)
        err = abs(x2 - x1)
        rows.append((i, x0, x1, x2, f0, f1, err))
        if abs(f(x2)) < tol or err < tol:
            return x2, err, i, rows
        x0, x1 = x1, x2
    return x2, err, max_iter, rows

def newton_raphson(f, df, x0, tol, max_iter):
    rows=[]
    x = x0
    for i in range(1, max_iter+1):
        fx = f(x); dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError("Zero derivative encountered in Newton-Raphson.")
        x_new = x - fx/dfx
        err = abs(x_new - x)
        rows.append((i, x, fx, dfx, x_new, err))
        if abs(fx) < tol or err < tol:
            return x_new, err, i, rows
        x = x_new
    return x, err, max_iter, rows

def fixed_point(g, x0, tol, max_iter):
    rows=[]
    x = x0
    for i in range(1, max_iter+1):
        x_new = g(x)
        err = abs(x_new - x)
        rows.append((i, x, x_new, err))
        if err < tol:
            return x_new, err, i, rows
        x = x_new
    return x, err, max_iter, rows

def modified_secant(f, x0, delta_frac, tol, max_iter):
    rows=[]
    x = x0
    for i in range(1, max_iter+1):
        delta = delta_frac * (abs(x) if x != 0 else 1.0)
        f_x = f(x)
        f_xd = f(x + delta)
        denom = f_xd - f_x
        if denom == 0:
            raise ZeroDivisionError("Zero denominator in modified secant.")
        x_new = x - f_x * (delta) / denom
        err = abs(x_new - x)
        rows.append((i, x, delta, f_x, f_xd, x_new, err))
        if abs(f_x) < tol or err < tol:
            return x_new, err, i, rows
        x = x_new
    return x, err, max_iter, rows

def prompt_float(prompt, default=None):
    while True:
        s = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if s=="" and default is not None:
            return default
        try:
            return float(s)
        except:
            print("Enter a numeric value.")

def main():
    print("Zero of Functions (ZOF) Solver — CLI")
    print("Enter function f(x) (e.g. x**3 - x - 2)")
    f_str = input("f(x) = ").strip()
    try:
        f, df, expr, df_expr = make_function(f_str)
    except Exception as e:
        print("Error parsing function:", e)
        sys.exit(1)

    tol = prompt_float("Tolerance (e.g. 1e-6)", 1e-6)
    max_iter = int(prompt_float("Max iterations", 50))

    methods = {
        '1': 'Bisection',
        '2': 'Regula Falsi',
        '3': 'Secant',
        '4': 'Newton-Raphson',
        '5': 'Fixed Point',
        '6': 'Modified Secant'
    }

    print("\nMethods:")
    for k,v in methods.items():
        print(f"{k}. {v}")
    choice = input("Choose method [1-6]: ").strip()
    try:
        if choice == '1':
            a = prompt_float("Left endpoint a")
            b = prompt_float("Right endpoint b")
            root, err, iters, rows = bisection(f, a, b, tol, max_iter)
            print("\nIteration | a        | b        | c        | f(a)     | f(b)     | f(c)     | error")
            for r in rows:
                print("{:9d} | {:8.6g} | {:8.6g} | {:8.6g} | {:8.6g} | {:8.6g} | {:8.6g} | {:8.6g}".format(*r))
        elif choice == '2':
            a = prompt_float("Left endpoint a")
            b = prompt_float("Right endpoint b")
            root, err, iters, rows = regula_falsi(f, a, b, tol, max_iter)
            print("\nIteration | a        | b        | c        | f(a)     | f(b)     | f(c)     | error")
            for r in rows:
                print("{:9d} | {:8.6g} | {:8.6g} | {:8.6g} | {:8.6g} | {:8.6g} | {:8.6g} | {:8.6g}".format(*r))
        elif choice == '3':
            x0 = prompt_float("x0 (first initial guess)")
            x1 = prompt_float("x1 (second initial guess)")
            root, err, iters, rows = secant(f, x0, x1, tol, max_iter)
            print("\nIter | x0       | x1       | x2       | f(x0)    | f(x1)    | error")
            for r in rows:
                print("{:4d} | {:8.6g} | {:8.6g} | {:8.6g} | {:8.6g} | {:8.6g} | {:8.6g}".format(*r))
        elif choice == '4':
            x0 = prompt_float("Initial guess x0")
            root, err, iters, rows = newton_raphson(f, df, x0, tol, max_iter)
            print("\nIter | x       | f(x)     | f'(x)    | x_new    | error")
            for r in rows:
                print("{:4d} | {:8.6g} | {:8.6g} | {:8.6g} | {:8.6g} | {:8.6g}".format(*r))
        elif choice == '5':
            print("Fixed Point: Either enter g(x) or choose automatic relaxation g(x)=x - lambda*f(x)")
            g_choice = input("Enter 'g' to provide g(x), or 'r' for relaxation [g/r]: ").strip().lower()
            if g_choice == 'g':
                g_str = input("g(x) = ").strip()
                x = sp.symbols('x')
                try:
                    g_expr = sp.sympify(g_str)
                    g = sp.lambdify(x, g_expr, 'numpy')
                except Exception as e:
                    print("Couldn't parse g(x):", e); return
                x0 = prompt_float("Initial guess x0")
                root, err, iters, rows = fixed_point(g, x0, tol, max_iter)
                print("\nIter | x_old    | x_new    | error")
                for r in rows:
                    print("{:4d} | {:8.6g} | {:8.6g} | {:8.6g}".format(*r))
            else:
                lam = prompt_float("Relaxation lambda (suggest 0.1)", 0.1)
                def g_rel(x): return x - lam * f(x)
                x0 = prompt_float("Initial guess x0")
                root, err, iters, rows = fixed_point(g_rel, x0, tol, max_iter)
                print("\nIter | x_old    | x_new    | error")
                for r in rows:
                    print("{:4d} | {:8.6g} | {:8.6g} | {:8.6g}".format(*r))
        elif choice == '6':
            x0 = prompt_float("Initial guess x0")
            delta = prompt_float("delta fraction (e.g. 1e-3)", 1e-3)
            root, err, iters, rows = modified_secant(f, x0, delta, tol, max_iter)
            print("\nIter | x       | delta    | f(x)     | f(x+delta) | x_new    | error")
            for r in rows:
                print("{:4d} | {:8.6g} | {:8.6g} | {:8.6g} | {:8.6g} | {:8.6g} | {:8.6g}".format(*r))
        else:
            print("Invalid choice.")
            return
    except Exception as e:
        print("Error during method:", e)
        return

    print("\nFinal result:")
    print(f"Estimated root: {root}")
    print(f"Final error estimate: {err}")
    print(f"Iterations: {iters}")

if __name__ == "__main__":
    main()
