import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.signal import find_peaks, savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
from sklearn.linear_model import TheilSenRegressor

# === 1. Load data ===
filename = r""
data = np.loadtxt(filename)

two_theta = data[:, 0]
intensity_scans = data[:, 1:]
intensity_avg = np.mean(intensity_scans, axis=1)

# === 2. Baseline (Asymmetric LS) ===
def baseline_als(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w, 0)
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

# === 3. Peak functions ===
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def lorentzian(x, a, x0, gamma):
    return a * gamma**2 / ((x - x0)**2 + gamma**2)

def voigt(x, a, x0, sigma, gamma):
    from scipy.special import wofz
    return a * np.real(wofz(((x - x0)+1j*gamma)/sigma)) / (sigma * np.sqrt(2*np.pi))

# === 4. Initial parameters ===
init_lam = 5e2
init_p = 0.04
init_peak_height = 0.09
init_distance = 10
vertical_offset = 0.15
lambda_x = 1.5406  # Å

# === 5. Auto-cleaning for Williamson–Hall (MEDIUM STRENGTH) ===
def clean_wh_points(x, y, max_iter=3):
    mask = np.ones(len(x), dtype=bool)
    for _ in range(max_iter):
        coeffs = np.polyfit(x[mask], y[mask], 1)
        m, c = coeffs
        y_fit = m * x + c
        residuals = np.abs(y - y_fit)

        threshold = 2 * np.median(residuals[mask])  # Medium rejection
        new_mask = residuals < threshold

        if new_mask.sum() == mask.sum():
            break
        mask = new_mask
    return mask

# === 6. Figure ===
fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.35, right=0.8)

line_avg, = ax.plot(two_theta, intensity_avg, alpha=0.5, label="Data")
background_line, = ax.plot(two_theta, np.zeros_like(two_theta), '--', color='gray', label="Background")
corrected_line, = ax.plot(two_theta, np.zeros_like(two_theta), color='blue', label="Corrected data")
peaks_line, = ax.plot([], [], 'x', color='black', label="Peaks")
peak_labels = []

ax.set_xlabel("2θ (°)")
ax.set_ylabel("Counts")
ax.set_title("XRD - Background & Peak Detection")
ax.legend()

# === 7. Sliders ===
axcolor = 'lightgoldenrodyellow'
ax_lam = plt.axes([0.15, 0.20, 0.55, 0.03], facecolor=axcolor)
ax_p = plt.axes([0.15, 0.15, 0.55, 0.03], facecolor=axcolor)
ax_height = plt.axes([0.15, 0.10, 0.55, 0.03], facecolor=axcolor)

slider_lam = Slider(ax_lam, "Lam", 1, 1e5, valinit=init_lam, valfmt="%1.0e")
slider_p = Slider(ax_p, "p", 0.001, 0.1, valinit=init_p, valfmt="%1.3f")
slider_height = Slider(ax_height, "Peak Height", 0.01, 2, valinit=init_peak_height)

# === 8. Fitting method selector ===
rax = plt.axes([0.82, 0.1, 0.12, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ("Gaussian", "Lorentzian", "Voigt"))
fit_method = "Voigt"

def select_method(label):
    global fit_method
    fit_method = label
radio.on_clicked(select_method)

# === 9. Regression method selector ===
rax_reg = plt.axes([0.82, 0.3, 0.12, 0.15], facecolor=axcolor)
radio_reg = RadioButtons(rax_reg, ("Weighted LS", "Theil-Sen"))
regression_method = "Theil-Sen"

def select_regression(label):
    global regression_method
    regression_method = label
radio_reg.on_clicked(select_regression)

# === 10. Manual peak add/remove ===
selected_peaks = set()

def on_click(event):
    if event.inaxes != ax or event.xdata is None:
        return

    x_click = event.xdata
    y_ref = intensity_corrected if "intensity_corrected" in globals() else intensity_avg

    idx_center = np.abs(two_theta - x_click).argmin()
    window = 5
    start = max(0, idx_center - window)
    end = min(len(y_ref), idx_center + window)
    pk = start + np.argmax(y_ref[start:end])

    if event.button == 1:  # add peak
        selected_peaks.add(pk)
        print(f"Added peak at {two_theta[pk]:.2f}")
    elif event.button == 3 and selected_peaks:  # remove nearest
        closest = min(selected_peaks, key=lambda i: abs(i - pk))
        selected_peaks.remove(closest)
        print(f"Removed peak near {two_theta[closest]:.2f}")

    update(None)

fig.canvas.mpl_connect("button_press_event", on_click)

# === 11. Update (core detection + fitting) ===
def update(val):
    global peaks, beta, theta, intensity_corrected, peak_labels

    lam = slider_lam.val
    p = slider_p.val
    peak_h = slider_height.val

    background = baseline_als(intensity_avg, lam=lam, p=p)
    intensity_corrected = intensity_avg - background

    y_smooth = savgol_filter(intensity_corrected, 15, 3)
    noise = np.std(y_smooth[:200])
    prominence = 3 * noise

    peaks_found, _ = find_peaks(
        y_smooth,
        height=np.max(y_smooth) * peak_h,
        distance=init_distance,
        prominence=prominence,
    )

    peaks = np.unique(np.concatenate((peaks_found, list(selected_peaks)))).astype(int)

    background_line.set_ydata(background)
    corrected_line.set_ydata(intensity_corrected)
    peaks_line.set_data(two_theta[peaks], intensity_corrected[peaks])

    for lbl in peak_labels:
        lbl.remove()
    peak_labels = []

    beta = []
    theta = []

    for pk in peaks:
        try:
            window = 25
            start = max(0, pk - window)
            end = min(len(two_theta), pk + window)
            xdat = two_theta[start:end]
            ydat = intensity_corrected[start:end]

            if fit_method == "Gaussian":
                popt, _ = curve_fit(gaussian, xdat, ydat,
                                    p0=[ydat.max(), two_theta[pk], 0.1])
                fwhm = 2.355 * abs(popt[2]) * np.pi / 180

            elif fit_method == "Lorentzian":
                popt, _ = curve_fit(lorentzian, xdat, ydat,
                                    p0=[ydat.max(), two_theta[pk], 0.1])
                fwhm = 2 * abs(popt[2]) * np.pi / 180

            else:
                popt, _ = curve_fit(
                    voigt, xdat, ydat,
                    p0=[ydat.max(), two_theta[pk], 0.1, 0.1],
                    maxfev=5000
                )
                fwhm = (0.5346 * 2 * abs(popt[3]) +
                        np.sqrt(0.2166*(2*abs(popt[3]))**2 +
                                (2.355*abs(popt[2]))**2)) * np.pi/180

            beta.append(fwhm)
            theta.append(np.radians(two_theta[pk]/2))

            txt = ax.text(two_theta[pk],
                          intensity_corrected[pk] + np.max(intensity_corrected) * vertical_offset,
                          f"{two_theta[pk]:.2f}",
                          rotation=90, fontsize=8, color="black")
            peak_labels.append(txt)

        except:
            continue

    beta = np.array(beta)
    theta = np.array(theta)
    print_peak_list()

    fig.canvas.draw_idle()


slider_lam.on_changed(update)
slider_p.on_changed(update)
slider_height.on_changed(update)

def print_peak_list():
    if len(peaks) == 0 or len(beta) == 0:
        return

    print("\n")  # spacing
    for i, pk in enumerate(peaks):
        if i >= len(beta):
            continue
        if np.isnan(beta[i]):
            continue

        fwhm_deg = beta[i] * (180 / np.pi)
        print(f"{two_theta[pk]:.6f},{fwhm_deg:.5f}")
    print("\n")


# === 12. Williamson–Hall ===
def williamson_hall(event=None):
    global beta, theta, peaks

    if len(beta) < 3:
        print("❌ Not enough peaks for W–H")
        return

    # --- compute raw x,y ---
    x_raw = 4 * np.sin(theta)
    y_raw = beta * np.cos(theta)

    # CLEAN GOOD PEAKS
    clean_mask = clean_wh_points(x_raw, y_raw)

    x = x_raw[clean_mask]
    y = y_raw[clean_mask]

    # PRINT PEAK LIST FOR PCRYSTALX
    print("\n===== Peak List for PCrystalX =====")

    # Important: use peak indices, not enumerate(two_theta)
    for i, pk in enumerate(peaks):

        if i >= len(beta):
            continue

        fwhm_rad = beta[i]
        if np.isnan(fwhm_rad):
            continue

        used = clean_mask[i] if i < len(clean_mask) else False
        status = "USED     " if used else "REJECTED "

        fwhm_deg = fwhm_rad * (180 / np.pi)

        print(f"{status}| 2θ = {two_theta[pk]:6.3f}°   |  FWHM = {fwhm_deg:8.5f}°")

    print("====================================\n")

    if len(x) < 3:
        print("❌ Not enough good peaks after cleaning")
        return


    # WH-FIT PLOT
    import matplotlib.pyplot as plt
    fig_wh, ax_wh = plt.subplots(figsize=(10, 6))

    # plot CLEANED peaks only
    ax_wh.scatter(x, y, color="blue", label="Data")


    # 1. Weighted LS
    if regression_method == "Weighted LS":

        weights = 1/(y**2 + 1e-12)
        coeffs, cov = np.polyfit(x, y, 1, w=weights, cov=True)

        slope = coeffs[0]
        intercept = coeffs[1]

        slope_err = np.sqrt(cov[0,0])
        intercept_err = np.sqrt(cov[1,1])

        strain = slope
        strain_err = slope_err

        size = (0.9 * lambda_x) / intercept
        size_err = (0.9 * lambda_x / intercept**2) * intercept_err

        y_fit = np.polyval(coeffs, x)
        fit_color = "red"


    # 2. Theil–Sen + Bootstrap
    else:
        N_BOOT = 1000
        slopes = []
        intercepts = []

        for _ in range(N_BOOT):
            idx = np.random.randint(len(x), size=len(x))
            xb = x[idx].reshape(-1,1)
            yb = y[idx]

            try:
                model = TheilSenRegressor().fit(xb, yb)
                slopes.append(model.coef_[0])
                intercepts.append(model.intercept_)
            except:
                pass

        slopes = np.array(slopes)
        intercepts = np.array(intercepts)

        # cleanup of invalid intercepts
        mask = np.isfinite(intercepts) & (intercepts > 0)
        slopes = slopes[mask]
        intercepts = intercepts[mask]

        # trim extremes
        if len(intercepts) > 20:
            lo, hi = np.percentile(intercepts, [5,95])
            m2 = (intercepts >= lo) & (intercepts <= hi)
            slopes = slopes[m2]
            intercepts = intercepts[m2]

        strain = np.mean(slopes)
        strain_err = np.std(slopes)

        sizes = (0.9 * lambda_x) / intercepts
        size = np.mean(sizes)
        size_err = np.std(sizes)

        y_fit = strain * x + (0.9 * lambda_x) / size
        fit_color = "green"

    # Plot fit
    ax_wh.plot(x, y_fit, color=fit_color, linewidth=2)

    ax_wh.set_xlabel("4 sin(θ)")
    ax_wh.set_ylabel("β cos(θ) [rad]")
    ax_wh.set_title(f"Williamson–Hall ({fit_method} FWHM)")

    # Dual legend system (text-left + method-right)
    from matplotlib.lines import Line2D

    # LEFT legend: ε, L
    legend_left = ax_wh.legend(
        handles=[
            Line2D([], [], color='none', label=f"ε = {strain:.2e} ± {strain_err:.1e}"),
            Line2D([], [], color='none', label=f"L = {size/10:.2f} ± {size_err/10:.2f} nm")
        ],
        loc='upper left',
        frameon=True,
        handlelength=0
    )
    ax_wh.add_artist(legend_left)

    # RIGHT legend: data + regression method
    legend_right = ax_wh.legend(
        handles=[
            Line2D([0],[0], marker='o', color='blue', linestyle='', label='Data'),
            Line2D([0],[0], color=fit_color, linestyle='-', label=regression_method)
        ],
        loc='upper right',
        frameon=True
    )

    # Print results
    print("\n===== Williamson–Hall results =====")
    print(f"Microstrain ε = {strain:.3e} ± {strain_err:.1e}")
    print(f"Crystallite size L = {size/10:.2f} ± {size_err/10:.2f} nm")
    print("===================================\n")

    plt.show()

# === 13. Scherrer Analysis ===
def scherrer_analysis(event=None):
    global beta, theta, peaks

    if len(beta) < 1:
        print("❌ No peaks available for Scherrer analysis")
        return

    # Convert FWHM from rad → already rad
    beta_rad = np.array(beta)   # ensure array

    # Scherrer formula: L = K λ / (β cosθ)
    K = 0.9
    lambda_A = lambda_x  # Å
    L_list_A = K * lambda_A / (beta_rad * np.cos(theta))  # Å
    L_list_nm = L_list_A / 10                              # nm

    # Filter out nonsense
    mask = np.isfinite(L_list_nm) & (L_list_nm > 0)
    L_list_nm = L_list_nm[mask]
    theta_plot = two_theta[peaks][mask]

    if len(L_list_nm) == 0:
        print("❌ No valid peaks for Scherrer (all invalid or zero β)")
        return

    L_mean = np.mean(L_list_nm)
    L_std = np.std(L_list_nm)

    # === Print results ===
    print("\n===== Scherrer Crystallite Sizes =====")
    for t, L in zip(theta_plot, L_list_nm):
        print(f"{t:.3f}°, {L:.3f} nm")
    print("--------------------------------------")
    print(f"Mean crystallite size: {L_mean:.3f} nm")
    print(f"Std dev: {L_std:.3f} nm")
    print("======================================\n")

    # === Plot ===
    fig_sch, ax_sch = plt.subplots(figsize=(8, 5))

    ax_sch.scatter(theta_plot, L_list_nm, color='purple')

    ax_sch.set_xlabel("2θ (°)")
    ax_sch.set_ylabel("Crystallite size L (nm)")
    ax_sch.set_title("Scherrer Crystallite Size per Peak")

    # text legend with mean + std
    from matplotlib.lines import Line2D
    leg = ax_sch.legend(
        handles=[
            Line2D([], [], color='none',
                   label=f"L_mean = {L_mean:.2f} nm"),
            Line2D([], [], color='none',
                   label=f"σ = {L_std:.2f} nm")
        ],
        frameon=True,
        loc="upper right",
        handlelength=0
    )
    ax_sch.add_artist(leg)

    plt.tight_layout()
    plt.show()


# === Button ===
ax_button = plt.axes([0.82, 0.015, 0.12, 0.04])
button_wh = Button(ax_button, "Williamson-Hall")
button_wh.on_clicked(williamson_hall)

# --- Button: Scherrer ---
ax_button_sch = plt.axes([0.82, 0.065, 0.12, 0.04])
button_sch = Button(ax_button_sch, "Scherrer Size")
button_sch.on_clicked(scherrer_analysis)


# === Initial plot ===
update(None)
plt.show()

