import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

def create_sequences(X, y, lb):
    return (np.array([X[i-lb:i] for i in range(lb,len(X))]),
            np.array([y[i]        for i in range(lb,len(X))]))

def evaluate_model(y_true, y_pred, name):
    # Dummy implementation, should be imported from evaluation.metrics
    return {'RMSE': 0, 'Directional Acc %': 0, 'P-Value': 0, 'Sig (p<0.05)': ''}

def build_and_train_lstm(X_lt, y_lt, X_le, y_le, lb, lstm_test_dates, predictions, metrics_list):
    lstm_model=Sequential([
        LSTM(128,return_sequences=True,input_shape=(lb,X_lt.shape[2])),
        BatchNormalization(),Dropout(0.3),
        LSTM(64,return_sequences=True),BatchNormalization(),Dropout(0.2),
        LSTM(32),Dropout(0.2),Dense(16,activation='relu'),Dense(1)
    ])
    lstm_model.compile(optimizer=Adam(1e-3),loss='huber',metrics=['mae'])
    lstm_model.summary()
    history=lstm_model.fit(X_lt,y_lt,validation_split=0.15,epochs=100,batch_size=32,shuffle=False,
        callbacks=[EarlyStopping(patience=5,restore_best_weights=True),
                   ReduceLROnPlateau(factor=0.5,patience=7,min_lr=1e-6)],verbose=1)
    y_pred_lstm=lstm_model.predict(X_le).flatten()
    predictions['LSTM']=(y_pred_lstm,lstm_test_dates)
    m_lstm=evaluate_model(y_le,y_pred_lstm,'LSTM'); metrics_list.append(m_lstm)
    print(f"LSTM: RMSE={m_lstm['RMSE']} DirAcc={m_lstm['Directional Acc %']}%")
    lstm_model.save('lstm_stock_model.keras')

    print("\n📊 STEP 7b: LSTM Training Curve")
    print("=" * 55)

    fig_lstm, axes_l = plt.subplots(1, 2, figsize=(14, 5))
    fig_lstm.suptitle('LSTM Training Diagnostics', fontsize=13, fontweight='bold')

    epochs_ran = range(1, len(history.history['loss']) + 1)

    # ── Panel 1: Loss curves ─────────────────────────────────────────────────
    ax_l = axes_l[0]
    ax_l.plot(epochs_ran, history.history['loss'],     color='#2196F3', linewidth=1.8, label='Train loss (Huber)')
    ax_l.plot(epochs_ran, history.history['val_loss'], color='#F44336', linewidth=1.8,
              linestyle='--', label='Val loss (Huber)')
    best_epoch = int(np.argmin(history.history['val_loss'])) + 1
    ax_l.axvline(best_epoch, color='#FFB300', linewidth=1.2, linestyle=':', label=f'Best epoch ({best_epoch})')
    ax_l.set_xlabel('Epoch'); ax_l.set_ylabel('Huber Loss')
    ax_l.set_title('Train vs Validation Loss'); ax_l.legend(fontsize=9); ax_l.grid(alpha=0.3)
    ax_l.text(0.02, 0.97, f'Epochs ran: {len(epochs_ran)}  |  Best: {best_epoch}',
              transform=ax_l.transAxes, fontsize=8, va='top', color='gray')

    # ── Panel 2: MAE curves ───────────────────────────────────────────────────
    ax_r = axes_l[1]
    ax_r.plot(epochs_ran, history.history['mae'],     color='#4CAF50', linewidth=1.8, label='Train MAE')
    ax_r.plot(epochs_ran, history.history['val_mae'], color='#FF9800', linewidth=1.8,
              linestyle='--', label='Val MAE')
    ax_r.axvline(best_epoch, color='#FFB300', linewidth=1.2, linestyle=':')
    ax_r.set_xlabel('Epoch'); ax_r.set_ylabel('MAE')
    ax_r.set_title('Train vs Validation MAE'); ax_r.legend(fontsize=9); ax_r.grid(alpha=0.3)

    # Overfitting flag
    gap = (np.array(history.history['val_loss']) - np.array(history.history['loss']))
    if gap[-1] > gap[0] * 2:
        axes_l[0].set_title('Train vs Validation Loss  ⚠️ overfit signal', color='#F44336', fontsize=10)

    plt.tight_layout()
    plt.savefig('lstm_training_curve.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Best val_loss = {min(history.history['val_loss']):.6f} at epoch {best_epoch}")
    print(f"  Final train/val gap = {gap[-1]:+.6f}  ({'⚠️  overfit' if gap[-1] > 0.001 else '✅  ok'})")
    print("  ✅ LSTM training curve saved → lstm_training_curve.png")

    return lstm_model, history, y_pred_lstm
