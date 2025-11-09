import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.signal import find_peaks, savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
from sklearn.linear_model import TheilSenRegressor

# === 1. Load data ===
filename = r"C:\Users\Florian\Desktop\Physik\Masterarbeit\Data\XRD\Floriank\Excess_Measurements\correct optics\Txt_Files\01102025_A2_Control_Excess_Measurement.txt"
data = np.loadtxt(filename)

two_theta = data[:, 0]
intensity_scans = data[:, 1:]
intensity_avg = np.mean(intensity_scans, axis=1)

# === 2. Asymmetric Least Squares Baseline ===
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

# === 3. Peak fitting functions ===
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def lorentzian(x, a, x0, gamma):
    return a * gamma**2 / ((x - x0) ** 2 + gamma**2)

def voigt(x, a, x0, sigma, gamma):
    from scipy.special import wofz
    return a * np.real(wofz(((x - x0) + 1j * gamma) / sigma)) / (sigma * np.sqrt(2 * np.pi))

# === 4. Initial Parameters ===
init_lam = 5e2
init_p = 0.04
init_peak_height = 0.09
init_distance = 10
vertical_offset = 0.15
lambda_x = 1.5406  # Cu Kα radiation (Å)

# === 5. Create Figure ===
fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.35, right=0.8)

line_avg, = ax.plot(two_theta, intensity_avg, label='Data', alpha=0.5)
background_line, = ax.plot(two_theta, np.zeros_like(two_theta), linestyle='--', color='gray', label='Background')
corrected_line, = ax.plot(two_theta, np.zeros_like(two_theta), color='blue', label='Corrected data')
peaks_line, = ax.plot([], [], "x", color='black', label='Peaks')
peak_labels = []

ax.set_xlabel("2θ (°)")
ax.set_ylabel("Counts")
ax.set_title("XRD - Background & Peak Detection")
ax.legend()

# === 6. Sliders ===
axcolor = 'lightgoldenrodyellow'
ax_lam = plt.axes([0.15, 0.20, 0.55, 0.03], facecolor=axcolor)
ax_p = plt.axes([0.15, 0.15, 0.55, 0.03], facecolor=axcolor)
ax_height = plt.axes([0.15, 0.10, 0.55, 0.03], facecolor=axcolor)

slider_lam = Slider(ax_lam, 'Lam', 1, 1e5, valinit=init_lam, valfmt='%1.0e')
slider_p = Slider(ax_p, 'p', 0.001, 0.1, valinit=init_p, valfmt='%1.3f')
slider_height = Slider(ax_height, 'Peak Height', 0.01, 2, valinit=init_peak_height)

# === 7. Fitting method menu ===
rax = plt.axes([0.82, 0.1, 0.12, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('Gaussian', 'Lorentzian', 'Voigt'))
fit_method = 'Voigt'

def select_method(label):
    global fit_method
    fit_method = label
radio.on_clicked(select_method)

# === 8. Regression method menu ===
rax_reg = plt.axes([0.82, 0.3, 0.12, 0.15], facecolor=axcolor)
radio_reg = RadioButtons(rax_reg, ('Weighted LS', 'Theil-Sen'))
regression_method = 'Theil-Sen'

def select_regression(label):
    global regression_method
    regression_method = label
radio_reg.on_clicked(select_regression)

# === 9. Interactive Peak Management ===
selected_peaks = set()  # store manually added peaks

def on_click(event):
    """Left-click = add peak; Right-click = remove nearest manual peak."""
    if event.inaxes != ax or event.xdata is None:
        return

    x_click = event.xdata

    # Use corrected data if available
    if 'intensity_corrected' in globals():
        y_ref = intensity_corrected
    else:
        y_ref = intensity_avg

    # Snap to local maximum after baseline correction
    idx_center = (np.abs(two_theta - x_click)).argmin()
    window = 5
    start = max(0, idx_center - window)
    end = min(len(y_ref), idx_center + window)
    local_max = start + np.argmax(y_ref[start:end])
    idx = local_max

    if event.button == 1:  # Left-click → add
        selected_peaks.add(idx)
        print(f"✅ Added peak at {two_theta[idx]:.2f}°")
    elif event.button == 3:  # Right-click → remove
        if not selected_peaks:
            return
        closest = min(selected_peaks, key=lambda i: abs(i - idx))
        selected_peaks.remove(closest)
        print(f"❌ Removed peak near {two_theta[closest]:.2f}°")

    # Force label redraw immediately after adding/removing
    update(None)

fig.canvas.mpl_connect('button_press_event', on_click)

# === 10. Update function ===
def update(val):
    global peak_labels, peaks, intensity_corrected, beta, theta
    lam = slider_lam.val
    p = slider_p.val
    peak_h = slider_height.val

    # --- Background subtraction ---
    background = baseline_als(intensity_avg, lam=lam, p=p)
    intensity_corrected = intensity_avg - background

    # --- Noise reduction for detection ---
    y_smooth = savgol_filter(intensity_corrected, window_length=15, polyorder=3)

    # --- Adaptive prominence based on noise ---
    noise_level = np.std(y_smooth[:200])
    prominence = 3 * noise_level

    # --- Peak detection ---
    peaks, _ = find_peaks(
        y_smooth,
        height=np.max(y_smooth) * peak_h,
        distance=init_distance,
        prominence=prominence
    )

    # Merge with manually added peaks
    peaks = np.unique(np.concatenate((peaks, list(selected_peaks)))).astype(int)

    # --- Update plots ---
    background_line.set_ydata(background)
    corrected_line.set_ydata(intensity_corrected)
    peaks_line.set_data(two_theta[peaks], intensity_corrected[peaks])

    # Remove old labels
    for label in peak_labels:
        label.remove()
    peak_labels = []

    # --- FWHM and annotation ---
    beta = []
    theta = []
    for pk in peaks:
        try:
            window = 25
            start = max(0, pk - window)
            end = min(len(two_theta), pk + window)
            x_data = two_theta[start:end]
            y_data = intensity_corrected[start:end]

            if fit_method == "Gaussian":
                p0 = [y_data.max(), two_theta[pk], 0.1]
                popt, _ = curve_fit(gaussian, x_data, y_data, p0=p0)
                fwhm = 2.355 * abs(popt[2]) * np.pi / 180
            elif fit_method == "Lorentzian":
                p0 = [y_data.max(), two_theta[pk], 0.1]
                popt, _ = curve_fit(lorentzian, x_data, y_data, p0=p0)
                fwhm = 2 * abs(popt[2]) * np.pi / 180
            else:  # Voigt
                p0 = [y_data.max(), two_theta[pk], 0.1, 0.1]
                popt, _ = curve_fit(voigt, x_data, y_data, p0=p0, maxfev=5000)
                fwhm = (0.5346 * 2 * abs(popt[3]) +
                        np.sqrt(0.2166 * (2 * abs(popt[3])) ** 2 +
                                (2.355 * abs(popt[2])) ** 2)) * np.pi / 180

            beta.append(fwhm)
            theta.append(np.radians(two_theta[pk] / 2))

            # Annotate peaks
            txt = ax.text(two_theta[pk],
                          intensity_corrected[pk] + np.max(intensity_corrected) * vertical_offset,
                          f"{two_theta[pk]:.2f}", rotation=90, verticalalignment='bottom',
                          fontsize=8, color='black')
            peak_labels.append(txt)


        except Exception as e:

            print(f"⚠️ Fit failed for peak at {two_theta[pk]:.2f}: {e}")

            # Still annotate manually added peaks even if fit fails

            txt = ax.text(two_theta[pk],

                          intensity_corrected[pk] + np.max(intensity_corrected) * vertical_offset,

                          f"{two_theta[pk]:.2f}", rotation=90,

                          verticalalignment='bottom', fontsize=8, color='black')

            peak_labels.append(txt)

    beta = np.array(beta)
    theta = np.array(theta)

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

slider_lam.on_changed(update)
slider_p.on_changed(update)
slider_height.on_changed(update)

# === 11. Williamson–Hall Analysis ===
def williamson_hall(event=None):
    if len(beta) < 3:
        print("❌ Not enough peaks for Williamson–Hall (need ≥ 3)")
        return

    x = 4 * np.sin(theta)
    y = beta * np.cos(theta)

    fig_wh, ax_wh = plt.subplots(figsize=(10, 6))
    ax_wh.scatter(x, y, color='blue', label="Data")

    if regression_method == "Weighted LS":
        weights = 1 / (y**2 + 1e-12)
        coeffs = np.polyfit(x, y, 1, w=weights)
        y_fit = np.polyval(coeffs, x)
        strain = coeffs[0]
        size = (0.9 * lambda_x) / coeffs[1]
        fit_color = 'red'
    else:
        model = TheilSenRegressor().fit(x.reshape(-1,1), y)
        y_fit = model.predict(x.reshape(-1,1))
        strain = model.coef_[0]
        size = (0.9 * lambda_x) / model.intercept_
        fit_color = 'green'

    ax_wh.plot(x, y_fit, color=fit_color, linestyle='-')
    ax_wh.set_xlabel("4 sin(θ)")
    ax_wh.set_ylabel("β cos(θ) [rad]")
    ax_wh.set_title(f"Williamson–Hall Plot ({fit_method} FWHM)")


    from matplotlib.lines import Line2D
    # === Upper-left legend with only text, no symbols ===
    from matplotlib.lines import Line2D

    # create invisible handles so matplotlib treats it as legend text but shows no marker/line
    empty_handle = Line2D([], [], color='none', label=f"ε = {strain:.2e}")
    empty_handle2 = Line2D([], [], color='none', label=f"L = {abs(size / 10):.1f} nm")

    legend1 = ax_wh.legend(handles=[empty_handle, empty_handle2],
                           loc='upper left', frameon=True, handlelength=0, handletextpad=0.2)
    ax_wh.add_artist(legend1)

    custom_lines = [
        Line2D([0],[0], color='blue', marker='o', linestyle='', label='Data'),
        Line2D([0],[0], color=fit_color, linestyle='-', label=regression_method)
    ]
    legend2 = ax_wh.legend(handles=custom_lines, loc='upper right', frameon=True)

    print(f"Microstrain (ε): {strain:.2e}")
    print(f"Crystallite size (L): {size/10:.2f} nm")
    plt.show()

ax_button = plt.axes([0.82, 0.015, 0.12, 0.04])
button_wh = Button(ax_button, "Williamson-Hall")
button_wh.on_clicked(williamson_hall)

# === 12. Initial Plot ===
update(None)
plt.show()
