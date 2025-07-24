#!/usr/bin/env python3
"""
Simple test page to verify location functionality
"""
import gradio as gr

def test_location_update(location_text):
    """Test function to verify location is being updated"""
    if location_text:
        return f"✅ Location received: {location_text}"
    else:
        return "❌ No location received"

def create_location_test():
    """Create a minimal test interface for location functionality"""
    
    location_js = """
    <script>
        function getLocation() {
            console.log('🔍 getLocation() called');
            
            if (navigator.geolocation) {
                console.log('✅ Geolocation supported');
                
                navigator.geolocation.getCurrentPosition(
                    function(position) {
                        console.log('✅ Position obtained:', position);
                        const lat = position.coords.latitude;
                        const lon = position.coords.longitude;
                        
                        console.log(`📍 Coordinates: ${lat}, ${lon}`);
                        
                        // Store location
                        const locationText = `${lat},${lon}`;
                        console.log('💾 Location stored:', locationText);
                        
                        // Update the gradio component
                        const locationInput = document.querySelector('#location-input input');
                        if (locationInput) {
                            console.log('✅ Location input found, updating...');
                            locationInput.value = locationText;
                            locationInput.dispatchEvent(new Event('input'));
                            console.log('✅ Location input updated');
                        } else {
                            console.error('❌ Location input not found');
                        }
                    },
                    function(error) {
                        console.error('❌ Geolocation error:', error);
                        alert('Unable to get your location: ' + error.message);
                    }
                );
            } else {
                console.error('❌ Geolocation not supported');
                alert('Geolocation not supported by this browser.');
            }
        }
    </script>
    """
    
    with gr.Blocks(head=location_js, title="Location Test") as app:
        gr.Markdown("# Location Test")
        
        with gr.Row():
            location_btn = gr.Button("📍 Get Location")
            location_input = gr.Textbox(label="Location", elem_id="location-input")
        
        location_output = gr.Textbox(label="Test Result", value="Waiting for location...")
        
        # Bind location button
        location_btn.click(
            fn=None,
            js="getLocation"
        )
        
        # Update output when location changes
        location_input.change(
            test_location_update,
            inputs=[location_input],
            outputs=[location_output]
        )
    
    return app

if __name__ == "__main__":
    app = create_location_test()
    app.launch(server_name="localhost", server_port=7861, share=False) 