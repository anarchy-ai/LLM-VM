from llm_vm.llm_vm_server.main import app


if __name__ == '__main__':
    # app.run(host="192.168.1.75",port=3002) # run at specified IP
    app.run(port=3002) # for running local
