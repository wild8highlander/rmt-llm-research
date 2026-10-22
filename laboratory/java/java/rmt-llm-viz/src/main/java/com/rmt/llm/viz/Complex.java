/*
 * Complex number class for Stieltjes transform and spectral analysis.
 *
 * Author: Iskhak Hamzatovich Isaev
 * ORCID:  0009-0003-7299-0701
 * License: CC-BY-NC-SA-4.0
 */

package com.rmt.llm.viz;

/**
 * Immutable complex number with basic arithmetic operations.
 */
public class Complex {
    public final double re;
    public final double im;

    public Complex(double re, double im) {
        this.re = re;
        this.im = im;
    }

    public Complex add(Complex other) {
        return new Complex(re + other.re, im + other.im);
    }

    public Complex subtract(Complex other) {
        return new Complex(re - other.re, im - other.im);
    }

    public Complex multiply(Complex other) {
        return new Complex(
            re * other.re - im * other.im,
            re * other.im + im * other.re
        );
    }

    public Complex divide(Complex other) {
        double denom = other.re * other.re + other.im * other.im;
        return new Complex(
            (re * other.re + im * other.im) / denom,
            (im * other.re - re * other.im) / denom
        );
    }

    public Complex negate() {
        return new Complex(-re, -im);
    }

    public Complex conjugate() {
        return new Complex(re, -im);
    }

    public double abs() {
        return Math.sqrt(re * re + im * im);
    }

    public double arg() {
        return Math.atan2(im, re);
    }

    /** Principal square root (branch with positive imaginary part). */
    public Complex sqrt() {
        double r = abs();
        double theta = arg() / 2;
        return new Complex(Math.sqrt(r) * Math.cos(theta), Math.sqrt(r) * Math.sin(theta));
    }

    @Override
    public String toString() {
        if (im == 0) return String.format("%.6f", re);
        if (re == 0) return String.format("%.6fi", im);
        return String.format("%.6f %s %.6fi", re, im >= 0 ? "+" : "-", Math.abs(im));
    }
}
