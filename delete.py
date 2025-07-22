import boto3
import time

sagemaker = boto3.client('sagemaker')

def delete_sagemaker_experiments():
    """
    Deletes all SageMaker Trials and then all Trial Components.
    """
    # --- 1. DELETE ALL TRIALS FIRST ---
    print("\n🧹 Starting SageMaker Trial cleanup...")
    try:
        paginator = sagemaker.get_paginator('list_trials')
        for page in paginator.paginate():
            for trial_summary in page.get('TrialSummaries', []):
                trial_name = trial_summary['TrialName']
                print(f"  - Deleting Trial: {trial_name}")
                try:
                    # Clean up associations first
                    for component_summary in sagemaker.list_trial_components(TrialName=trial_name)['TrialComponentSummaries']:
                         sagemaker.disassociate_trial_component(TrialName=trial_name, TrialComponentName=component_summary['TrialComponentName'])
                    
                    sagemaker.delete_trial(TrialName=trial_name)
                except Exception as e:
                    print(f"    ❌ ERROR deleting trial {trial_name}: {str(e)}")
    except Exception as e:
        print(f"Could not list or delete trials: {e}")

    # --- 2. WAIT FOR DELETION TO PROCESS ---
    print("\n⏳ Waiting 30 seconds for AWS to process trial deletions...")
    time.sleep(30)

    # --- 3. DELETE ALL TRIAL COMPONENTS ---
    print("\n🧹 Starting SageMaker Trial Component cleanup...")
    try:
        paginator = sagemaker.get_paginator('list_trial_components')
        for page in paginator.paginate():
            for summary in page.get('TrialComponentSummaries', []):
                name = summary['TrialComponentName']
                print(f"  - Deleting Trial Component: {name}")
                try:
                    sagemaker.delete_trial_component(TrialComponentName=name)
                except Exception as e:
                    print(f"    ❌ ERROR deleting component {name}: {str(e)}")
    except Exception as e:
         print(f"Could not list or delete trial components: {e}")


def delete_endpoints():
    """
    Finds and deletes all SageMaker Endpoints and their associated configs.
    """
    print("\n🗑️ Starting endpoint cleanup...")

    paginator = sagemaker.get_paginator('list_endpoints')
    endpoints_found = 0
    
    for page in paginator.paginate():
        for ep in page.get('Endpoints', []):
            endpoints_found += 1
            ep_name = ep['EndpointName']
            ep_status = ep['EndpointStatus']
            print(f"Found endpoint: {ep_name} (status: {ep_status})")

            if ep_status != 'Deleting':
                try:
                    ep_config_name = sagemaker.describe_endpoint(EndpointName=ep_name)['EndpointConfigName']
                    sagemaker.delete_endpoint(EndpointName=ep_name)
                    print(f"  - Deleting endpoint: {ep_name}")
                    sagemaker.delete_endpoint_config(EndpointConfigName=ep_config_name)
                    print(f"  - Deleting endpoint config: {ep_config_name}")
                except Exception as e:
                    print(f"  ❌ ERROR deleting {ep_name}: {str(e)}")
    
    if endpoints_found == 0:
        print("No active endpoints found.")


if __name__ == "__main__":
    delete_sagemaker_experiments()
    delete_endpoints()
    print("\n✅ SageMaker cleanup completed.")