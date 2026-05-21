import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

st.set_page_config(
    page_title="Dashboard de Ventas",
    layout="wide"
)

st.title("Dashboard de Ventas — Tiendas")
st.markdown("---")

# Carga de los datos
@st.cache_data
def load_data(file_name):
    df = pd.read_excel(file_name)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    col_map = {
        "fecha": "fecha", "tienda": "tienda",
        "categoria_de_producto": "categoria", "categoria": "categoria",
        "vendedor": "vendedor", "producto": "producto",
        "cantidad": "cantidad", "precio": "precio", "total": "total",
    }
    df.rename(columns={c: col_map[c] for c in df.columns if c in col_map}, inplace=True)
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=False, errors="coerce")
    
    # Componentes para el modelo
    df["año"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    df["dia_semana_num"] = df["fecha"].dt.dayofweek
    
    return df

# Se define únicamente el nombre del archivo (debe estar en la misma carpeta que el script)
FILE_NAME = "Ventas_Tienda.xlsx"

try:
    df_f = load_data(FILE_NAME)
except FileNotFoundError:
    st.error(f"No se encontró el archivo `{FILE_NAME}` en el directorio actual.\n\n"
             f"Asegúrate de colocar el archivo Excel exactamente en la misma carpeta donde estás ejecutando este script de Streamlit.")
    st.stop()

# Métricas principales (calculadas con la totalidad de los datos)
total_ventas   = df_f["total"].sum()
total_unidades = df_f["cantidad"].sum()
ticket_prom    = df_f["total"].mean()
num_transac    = len(df_f)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Ventas Totales",     f"${total_ventas:,.2f}")
k2.metric("Unidades Vendidas", f"{total_unidades:,}")
k3.metric("Ticket Promedio",   f"${ticket_prom:,.2f}")
k4.metric("Transacciones",     f"{num_transac}")

st.markdown("---")

# Pestañas de análisis
tab_historico, tab_estadisticas, tab_prediccion = st.tabs([
    "Análisis Histórico General", 
    "Estadísticas por Día", 
    "Modelo Predictivo (3 Años)"
])

# Pestaña 1: Análisis Histórico General
with tab_historico:
    col_yoy, col_store = st.columns(2)
    with col_yoy:
        st.subheader("Comparativa de Porcentajes por Año ")
        ventas_anuales = df_f.groupby("año")["total"].sum().reset_index()
        ventas_anuales["pct_cambio"] = ventas_anuales["total"].pct_change() * 100
        
        if len(ventas_anuales) > 1:
            ultimo_anual = ventas_anuales.iloc[-1]
            st.metric(
                label=f"Variación del último año ({int(ultimo_anual['año'])})", 
                value=f"${ultimo_anual['total']:,.2f}", 
                delta=f"{ultimo_anual['pct_cambio']:+.2f}%"
            )
        else:
            st.info("Se requieren datos de al menos 2 años para calcular el porcentaje de aumento o disminución.")
            
        fig_yoy = px.bar(ventas_anuales, x="año", y="total", text_auto=',.0f',
                         labels={"total": "Ventas ($)", "año": "Año"},
                         color_discrete_sequence=["#1F77B4"])
        fig_yoy.update_layout(plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(dtick=1))
        st.plotly_chart(fig_yoy, use_container_width=True)

    with col_store:
        st.subheader("Ventas por Tienda")
        tienda_df = df_f.groupby("tienda")["total"].sum().reset_index().sort_values("total", ascending=False)
        fig = px.bar(tienda_df, x="tienda", y="total", color="tienda",
                     labels={"total": "Ventas ($)", "tienda": "Tienda"},
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    col_cat, col_vend = st.columns(2)
    with col_cat:
        st.subheader("Ventas por Categoría")
        cat_df = df_f.groupby("categoria")["total"].sum().reset_index()
        fig = px.pie(cat_df, names="categoria", values="total", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_vend:
        st.subheader(" Rendimiento por Vendedor")
        vend_df = df_f.groupby("vendedor").agg(Ventas=("total", "sum")).reset_index().sort_values("Ventas", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=vend_df["vendedor"], x=vend_df["Ventas"],
            orientation="h", name="Ventas ($)",
            marker_color=px.colors.qualitative.Set1[1],
            text=vend_df["Ventas"].apply(lambda v: f"${v:,.0f}"),
            textposition="outside"
        ))
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Ventas ($)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(" Evolución Diaria de Ventas")
    daily_df = df_f.groupby("fecha")["total"].sum().reset_index()
    fig = px.line(daily_df, x="fecha", y="total", markers=True,
                  labels={"total": "Ventas ($)", "fecha": "Fecha"},
                  color_discrete_sequence=["#636EFA"])
    fig.update_traces(line_width=2, marker_size=7)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Productos por Ventas")
    prod_df = df_f.groupby("producto")["total"].sum().reset_index().sort_values("total", ascending=False).head(10)
    fig = px.bar(prod_df, x="total", y="producto", orientation="h",
                 labels={"total": "Ventas ($)", "producto": "Producto"},
                 color="total", color_continuous_scale="Teal")
    fig.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# Pestaña 2: Estadísticas por Día
with tab_estadisticas:
    st.subheader("Análisis Estadístico y Promedios por Día de la Semana")
    
    dias_semana = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}
    
    stats_dias = df_f.groupby("dia_semana_num").agg(
        ventas_totales=("total", "sum"),
        ticket_promedio=("total", "mean"),
        unidades_promedio=("cantidad", "mean"),
        transacciones=("total", "count")
    ).reset_index()
    stats_dias["dia_nombre"] = stats_dias["dia_semana_num"].map(dias_semana)
    stats_dias = stats_dias.sort_values("dia_semana_num")

    col_t1, col_t2 = st.columns([1.3, 2])
    
    with col_t1:
        st.markdown("**Tabla resumen de ocurrencias:**")
        
        resumen_vista = pd.DataFrame()
        resumen_vista["Día"] = stats_dias["dia_nombre"]
        resumen_vista["N° Transacciones"] = stats_dias["transacciones"].astype(int)
        resumen_vista["Ticket Promedio ($)"] = stats_dias["ticket_promedio"].apply(lambda x: f"${x:,.2f}")
        resumen_vista["Cant. Promedio"] = stats_dias["unidades_promedio"].round(1)
        
        st.dataframe(
            resumen_vista.set_index("Día"), 
            use_container_width=True
        )
        
    with col_t2:
        fig_dias = px.bar(stats_dias, x="dia_nombre", y="ventas_totales",
                          title="Volumen Acumulado de Ventas por Día",
                          labels={"ventas_totales": "Ventas Totales ($)", "dia_nombre": "Día de la Semana"},
                          color="ventas_totales", color_continuous_scale="Viridis")
        fig_dias.update_layout(plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig_dias, use_container_width=True)

# Pestaña 3: Modelo Predictivo
with tab_prediccion:
    st.subheader(" Proyección de Ventas del Modelo")
    
    df_pred = df_f.groupby("año")["total"].sum().reset_index()
    
    if len(df_pred) >= 2:
        X = df_pred["año"].values
        y = df_pred["total"].values
        
        m, b = np.polyfit(X, y, 1)
        
        ultimo_año = int(X[-1])
        ultimo_valor_real = y[-1]
        años_futuros = np.array([ultimo_año + 1, ultimo_año + 2, ultimo_año + 3])
        ventas_futuras = m * años_futuros + b
        ventas_futuras = np.clip(ventas_futuras, a_min=0, a_max=None)
        
        df_hist = pd.DataFrame({"Año": X, "Ventas ($)": y, "Tipo": "Histórico"})
        df_futuro = pd.DataFrame({"Año": años_futuros, "Ventas ($)": ventas_futuras, "Tipo": "Predicción"})
        df_total_modelo = pd.concat([df_hist, df_futuro], ignore_index=True)
        
        col_p1, col_p2 = st.columns([2, 1])
        
        with col_p1:
            fig_line_pred = px.line(df_total_modelo, x="Año", y="Ventas ($)", color="Tipo", markers=True,
                                    title="Tendencia Histórica vs Predicción para los Próximos 3 Años",
                                    color_discrete_map={"Histórico": "#636EFA", "Predicción": "#EF553B"})
            fig_line_pred.update_layout(plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(dtick=1))
            fig_line_pred.update_traces(line_width=3, marker_size=8)
            st.plotly_chart(fig_line_pred, use_container_width=True)
            
        with col_p2:
            st.markdown("**Valores calculados por el modelo:**")
            df_futuro_display = df_futuro[["Año", "Ventas ($)"]].copy()
            df_futuro_display["Ventas ($)"] = df_futuro_display["Ventas ($)"].apply(lambda v: f"${v:,.2f}")
            st.dataframe(df_futuro_display.set_index("Año"), use_container_width=True)

        st.markdown("---")
        st.markdown("###  ¿Qué pasará en los próximos 3 años? ")
        
        cambio_proyectado = ((ventas_futuras[-1] - ultimo_valor_real) / ultimo_valor_real) * 100
        año_final = años_futuros[-1]
        
        if m > 0:
            st.success(f"""
            **Tendencia de Crecimiento:** El modelo predictivo indica que las ventas **aumentarán** de forma sostenida durante los próximos 3 años.
            
            * **Impacto estimado:** Se proyecta un incremento del **{cambio_proyectado:.2f}%** para el año **{año_final}** en comparación con el cierre de **{ultimo_año}**.
            * **Pico de Ventas:** El año con mayores ingresos estimados será **{año_final}** alcanzando una cifra aproximada de **${ventas_futuras[-1]:,.2f}**.
            """)
        elif m < 0:
            st.warning(f"""
            **Tendencia de Disminución:** El comportamiento histórico indica una desaceleración en las ventas. Si las condiciones actuales se mantienen sin cambios estratégicos, el modelo proyecta que las ventas **disminuirán**.
            
            * **Impacto estimado:** Se proyecta una reducción del **{abs(cambio_proyectado):.2f}%** para el año **{año_final}** con respecto al año **{ultimo_año}**.
            * **Sugerencia:** Se recomienda revisar el rendimiento global de la estrategia comercial para contrarrestar esta tendencia.
            """)
        else:
            st.info(f"**Tendencia Estable:** El modelo predice que las ventas se mantendrán completamente estables, sin variaciones significativas en los años {años_futuros[0]}, {años_futuros[1]} y {años_futuros[2]}.")
            
    else:
        st.info(" Se necesita un registro histórico mínimo de 2 años diferentes para construir la línea de tendencia y estimar los próximos 3 años.")

# Tabla detallada (Final)
st.markdown("---")
st.subheader("Detalle de Transacciones")
st.dataframe(
    df_f.sort_values("fecha", ascending=False).reset_index(drop=True),
    use_container_width=True,
    height=300
)