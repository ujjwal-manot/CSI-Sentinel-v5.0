idf_component_register(
    SRCS "main.c" "wifi_manager.c" "csi_collector.c" "udp_transmitter.c"
    INCLUDE_DIRS "."
    REQUIRES esp_wifi nvs_flash esp_netif lwip esp_timer
)
